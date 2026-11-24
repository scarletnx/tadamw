/* SPDX-License-Identifier: GPL-3.0-or-later */
/* GPU trainer for the profile 1 mixer (docs/design.md, "Encoder pipeline").
 *
 * Vulkan is loaded at run time (dlopen of libvulkan.so.1), so the library
 * has no link-time dependency on it and runs where there is no loader or
 * GPU; the caller then fits on the CPU. Two compute shaders (compiled to
 * SPIR-V at build time and embedded) train several candidates at once with
 * the CPU trainer's schedule: fit_gpu_prep.comp sorts the batches of the
 * next few hundred steps by weight set, all in parallel, and fit_gpu.comp
 * then runs those steps, one workgroup per candidate. A whole fit is one
 * command buffer of such dispatch pairs. */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fit_internal.h"
#include "taw_internal.h"

/* (also without GPU support, so that TAW_GPU_ON fails the same way) */
bool taw_gpu_fits(const p1_shape *s)
{
	return s->hidden <= 16; /* fit_gpu.comp's HM */
}

/* The shaders, layout_of() and pad_rows() below all assume the profile-1
 * row width P1_F (26). Profile 14 rows are P1_F_HNET (27) wide; passing one
 * of its nets here would silently copy/read with the wrong stride. The
 * encoder already keeps profile 14 on the CPU; reject it here too so the
 * hazard cannot be reached from another caller. */
static bool gpu_net_fits(const tnet *t)
{
	return t->f == P1_F;
}

#if TAW_GPU

#include <dlfcn.h>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include "fit_gpu_prep_spv.h"
#include "fit_gpu_spv.h"

enum {
	GPU_NB = 256,         /* batch (= FIT_BATCH) */
	GPU_PARTS = 2,        /* fit_gpu.comp: threads per sample */
	GPU_THREADS = GPU_NB * GPU_PARTS, /* its workgroup size */
	GPU_SW = 14,          /* words per sample */
	GPU_DESC = 4096,      /* meta: squash table, then descriptors */
	GPU_CHUNK = 256,      /* steps per dispatch */
	GPU_BLK = 1088,       /* words per prepared batch */
	GPU_SHARED = 46080,   /* shared memory fit_gpu.comp needs (45200), rounded up */
};
static_assert((int)GPU_NB == (int)FIT_BATCH, "the CPU trainer's batch");
#define GPU_STAGE ((VkDeviceSize)32 << 20) /* staging buffer bytes */
/* VRAM we allow ourselves: the GPU is likely shared with a desktop. */
#define GPU_VRAM_MAX ((VkDeviceSize)600 << 20)

typedef struct {
	uint32_t mode, nbatch, last_nb, step0, steps_total, ns, stride, step_nb,
	    tied, pos_off, ncand;
} gpu_push;

#define VK_INST_FNS(X)                                                      \
	X(DestroyInstance) X(EnumeratePhysicalDevices)                      \
	X(GetPhysicalDeviceProperties) X(GetPhysicalDeviceQueueFamilyProperties) \
	X(GetPhysicalDeviceMemoryProperties) X(CreateDevice) X(GetDeviceProcAddr) \
	X(GetPhysicalDeviceProperties2)
#define VK_DEV_FNS(X)                                                       \
	X(DestroyDevice) X(GetDeviceQueue) X(CreateBuffer) X(DestroyBuffer)  \
	X(GetBufferMemoryRequirements) X(AllocateMemory) X(FreeMemory)       \
	X(BindBufferMemory) X(MapMemory) X(CreateShaderModule)               \
	X(DestroyShaderModule) X(CreateDescriptorSetLayout)                  \
	X(DestroyDescriptorSetLayout) X(CreatePipelineLayout)                \
	X(DestroyPipelineLayout) X(CreateComputePipelines) X(DestroyPipeline) \
	X(CreateDescriptorPool) X(DestroyDescriptorPool)                     \
	X(AllocateDescriptorSets) X(UpdateDescriptorSets) X(CreateCommandPool) \
	X(DestroyCommandPool) X(AllocateCommandBuffers) X(BeginCommandBuffer) \
	X(EndCommandBuffer) X(ResetCommandBuffer) X(CmdBindPipeline)         \
	X(CmdBindDescriptorSets) X(CmdPushConstants) X(CmdDispatch)          \
	X(CmdPipelineBarrier) X(CmdCopyBuffer) X(QueueSubmit) X(CreateFence) \
	X(DestroyFence) X(WaitForFences) X(ResetFences)

typedef struct {
	VkBuffer b;
	VkDeviceMemory m;
	void *map;
	VkDeviceSize size;
} gbuf;

struct taw_gpu {
	void *lib;
	bool debug;
	PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
	PFN_vkCreateInstance CreateInstance;
#define X(n) PFN_vk##n n;
	VK_INST_FNS(X)
	VK_DEV_FNS(X)
#undef X
	VkInstance inst;
	VkPhysicalDevice pd;
	VkDevice dev;
	VkQueue q;
	uint32_t qf;
	VkPhysicalDeviceMemoryProperties mp;
	char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
	gbuf smp, stage;
	VkDescriptorSetLayout dsl;
	VkPipelineLayout pl;
	VkPipeline pipe, prep;
	VkDescriptorPool dp;
	VkDescriptorSet ds;
	VkCommandPool cp;
	VkCommandBuffer cb;
	VkFence fence;
	size_t ns;
};

[[gnu::format(printf, 2, 3)]] static void note(const taw_gpu *g,
    const char *fmt, ...)
{
	if (!g->debug)
		return;
	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr, "taw fit: gpu: ");
	vfprintf(stderr, fmt, ap);
	fputc('\n', stderr);
	va_end(ap);
}

#define CK(call)                                                            \
	do {                                                                \
		VkResult r_ = (call);                                       \
		if (r_ != VK_SUCCESS) {                                     \
			note(g, "%s failed (%d)", #call, (int)r_);          \
			return false;                                       \
		}                                                           \
	} while (0)

static int memtype(const taw_gpu *g, uint32_t bits, VkMemoryPropertyFlags want,
    VkMemoryPropertyFlags avoid)
{
	for (uint32_t i = 0; i < g->mp.memoryTypeCount; i++)
		if ((bits & (1u << i)) &&
		    (g->mp.memoryTypes[i].propertyFlags & want) == want &&
		    !(g->mp.memoryTypes[i].propertyFlags & avoid))
			return (int)i;
	return -1;
}

/* host: a mapped staging buffer; else device-local storage. */
static bool mkbuf(taw_gpu *g, VkDeviceSize size, bool host, gbuf *out)
{
	*out = (gbuf){.size = size};
	VkBufferCreateInfo bi = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = host ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
		    VK_BUFFER_USAGE_TRANSFER_DST_BIT
			      : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
		    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
		    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	CK(g->CreateBuffer(g->dev, &bi, nullptr, &out->b));
	VkMemoryRequirements mr;
	g->GetBufferMemoryRequirements(g->dev, out->b, &mr);
	int t;
	if (host) {
		const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		t = memtype(g, mr.memoryTypeBits,
		    hv | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
		if (t < 0)
			t = memtype(g, mr.memoryTypeBits, hv, 0);
	} else {
		t = memtype(g, mr.memoryTypeBits,
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
		if (t < 0)
			t = memtype(g, mr.memoryTypeBits,
			    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
	}
	if (t < 0) {
		note(g, "no memory type for a %s buffer",
		    host ? "staging" : "device");
		return false;
	}
	VkMemoryAllocateInfo ai = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = mr.size,
		.memoryTypeIndex = (uint32_t)t,
	};
	CK(g->AllocateMemory(g->dev, &ai, nullptr, &out->m));
	CK(g->BindBufferMemory(g->dev, out->b, out->m, 0));
	if (host)
		CK(g->MapMemory(g->dev, out->m, 0, VK_WHOLE_SIZE, 0, &out->map));
	return true;
}

static void rmbuf(taw_gpu *g, gbuf *b)
{
	if (b->b)
		g->DestroyBuffer(g->dev, b->b, nullptr);
	if (b->m)
		g->FreeMemory(g->dev, b->m, nullptr); /* unmaps */
	*b = (gbuf){0};
}

static bool begin(taw_gpu *g)
{
	CK(g->ResetCommandBuffer(g->cb, 0));
	VkCommandBufferBeginInfo bi = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	CK(g->BeginCommandBuffer(g->cb, &bi));
	return true;
}

static bool submit_wait(taw_gpu *g)
{
	CK(g->EndCommandBuffer(g->cb));
	CK(g->ResetFences(g->dev, 1, &g->fence));
	VkSubmitInfo si = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &g->cb,
	};
	CK(g->QueueSubmit(g->q, 1, &si, g->fence));
	CK(g->WaitForFences(g->dev, 1, &g->fence, VK_TRUE, UINT64_MAX));
	return true;
}

/* Copy between host memory and a device buffer through the staging buffer.
 * fill(ctx, dst, off, n), when given, produces the upload bytes instead of
 * src. */
typedef void (*fill_fn)(void *ctx, void *dst, size_t off, size_t n);

static bool xfer(taw_gpu *g, VkBuffer dev, bool up, void *host, size_t n,
    fill_fn fill, void *ctx)
{
	/* whole samples per round */
	const size_t round = (size_t)(GPU_STAGE - GPU_STAGE % (GPU_SW * 4));
	for (size_t off = 0; off < n;) {
		size_t len = n - off < round ? n - off : round;
		if (up) {
			if (fill)
				fill(ctx, g->stage.map, off, len);
			else
				memcpy(g->stage.map, (uint8_t *)host + off, len);
		}
		if (!begin(g))
			return false;
		VkBufferCopy bc = {.srcOffset = up ? 0 : off,
			.dstOffset = up ? off : 0, .size = len};
		if (up)
			g->CmdCopyBuffer(g->cb, g->stage.b, dev, 1, &bc);
		else
			g->CmdCopyBuffer(g->cb, dev, g->stage.b, 1, &bc);
		if (!submit_wait(g))
			return false;
		if (!up)
			memcpy((uint8_t *)host + off, g->stage.map, len);
		off += len;
	}
	return true;
}

static bool load_device(taw_gpu *g)
{
#define X(n)                                                                \
	if (!(g->n = (PFN_vk##n)g->GetDeviceProcAddr(g->dev, "vk" #n))) {    \
		note(g, "no vk%s", #n);                              \
		return false;                                               \
	}
	VK_DEV_FNS(X)
#undef X
	return true;
}

static bool pick_device(taw_gpu *g)
{
	VkApplicationInfo app = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "tadamw",
		.apiVersion = VK_API_VERSION_1_1,
	};
	VkInstanceCreateInfo ici = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &app,
	};
	CK(g->CreateInstance(&ici, nullptr, &g->inst));
#define X(n)                                                                \
	if (!(g->n = (PFN_vk##n)g->GetInstanceProcAddr(g->inst, "vk" #n))) { \
		note(g, "no vk%s", #n);                              \
		return false;                                               \
	}
	VK_INST_FNS(X)
#undef X
	VkPhysicalDevice pds[16];
	uint32_t npd = 16;
	VkResult r = g->EnumeratePhysicalDevices(g->inst, &npd, pds);
	if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || npd == 0) {
		note(g, "no devices (%d)", (int)r);
		return false;
	}
	/* TAW_GPU_DEVICE=n picks device n, even a CPU implementation (for
	 * testing); by default the best GPU: discrete, integrated, virtual. */
	const char *want = getenv("TAW_GPU_DEVICE");
	int best = -1, score = 0;
	for (uint32_t i = 0; i < npd; i++) {
		VkPhysicalDeviceProperties pp;
		g->GetPhysicalDeviceProperties(pds[i], &pp);
		int s;
		if (want && *want)
			s = (uint32_t)atoi(want) == i ? 1 : 0;
		else
			s = pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 4
			    : pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 3
			    : pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU ? 2
			    : 0;
		const VkPhysicalDeviceLimits *L = &pp.limits;
		if (L->maxComputeSharedMemorySize < GPU_SHARED ||
		    L->maxComputeWorkGroupInvocations < GPU_THREADS ||
		    L->maxComputeWorkGroupSize[0] < GPU_THREADS ||
		    L->maxStorageBufferRange < (1u << 28) ||
		    L->maxComputeWorkGroupCount[0] < GPU_CHUNK)
			s = 0;
		/* the trainer adds a sample's parts with subgroup shuffles */
		VkPhysicalDeviceSubgroupProperties sp = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
		VkPhysicalDeviceProperties2 pp2 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
			.pNext = &sp};
		g->GetPhysicalDeviceProperties2(pds[i], &pp2);
		if (pp.apiVersion < VK_API_VERSION_1_1 || sp.subgroupSize < GPU_PARTS ||
		    !(sp.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ||
		    !(sp.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT))
			s = 0;
		if (s > score) {
			score = s;
			best = (int)i;
			memcpy(g->name, pp.deviceName, sizeof g->name);
			g->name[sizeof g->name - 1] = 0;
		}
	}
	if (best < 0) {
		note(g, "no suitable device (%u found)", npd);
		return false;
	}
	g->pd = pds[best];
	g->GetPhysicalDeviceMemoryProperties(g->pd, &g->mp);
	VkQueueFamilyProperties qfp[16];
	uint32_t nqf = 16;
	g->GetPhysicalDeviceQueueFamilyProperties(g->pd, &nqf, qfp);
	g->qf = UINT32_MAX;
	for (uint32_t i = 0; i < nqf; i++)
		if ((qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
		    (g->qf == UINT32_MAX ||
			!(qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)))
			g->qf = i; /* prefer a compute-only queue */
	if (g->qf == UINT32_MAX) {
		note(g, "%s has no compute queue", g->name);
		return false;
	}
	const float prio = 1.0f;
	VkDeviceQueueCreateInfo qi = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = g->qf,
		.queueCount = 1,
		.pQueuePriorities = &prio,
	};
	VkDeviceCreateInfo dci = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &qi,
	};
	CK(g->CreateDevice(g->pd, &dci, nullptr, &g->dev));
	if (!load_device(g))
		return false;
	g->GetDeviceQueue(g->dev, g->qf, 0, &g->q);
	return true;
}

static bool mkpipe(taw_gpu *g, const uint32_t *code, size_t size,
    VkPipeline *out)
{
	VkShaderModuleCreateInfo si = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = code,
	};
	VkShaderModule sm;
	CK(g->CreateShaderModule(g->dev, &si, nullptr, &sm));
	VkComputePipelineCreateInfo ci = {
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = sm,
			.pName = "main",
		},
		.layout = g->pl,
	};
	VkResult r = g->CreateComputePipelines(g->dev, VK_NULL_HANDLE, 1, &ci,
	    nullptr, out);
	g->DestroyShaderModule(g->dev, sm, nullptr);
	CK(r);
	return true;
}

static bool make_pipeline(taw_gpu *g)
{
	VkDescriptorSetLayoutBinding lb[4];
	for (uint32_t i = 0; i < 4; i++)
		lb[i] = (VkDescriptorSetLayoutBinding){
			.binding = i,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		};
	VkDescriptorSetLayoutCreateInfo lci = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 4,
		.pBindings = lb,
	};
	CK(g->CreateDescriptorSetLayout(g->dev, &lci, nullptr, &g->dsl));
	VkPushConstantRange pr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.size = sizeof(gpu_push)};
	VkPipelineLayoutCreateInfo plci = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &g->dsl,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pr,
	};
	CK(g->CreatePipelineLayout(g->dev, &plci, nullptr, &g->pl));
	if (!mkpipe(g, fit_gpu_spv, sizeof fit_gpu_spv, &g->pipe) ||
	    !mkpipe(g, fit_gpu_prep_spv, sizeof fit_gpu_prep_spv, &g->prep))
		return false;
	VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 4};
	VkDescriptorPoolCreateInfo dpci = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1,
		.poolSizeCount = 1,
		.pPoolSizes = &ps,
	};
	CK(g->CreateDescriptorPool(g->dev, &dpci, nullptr, &g->dp));
	VkDescriptorSetAllocateInfo dsai = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = g->dp,
		.descriptorSetCount = 1,
		.pSetLayouts = &g->dsl,
	};
	CK(g->AllocateDescriptorSets(g->dev, &dsai, &g->ds));
	VkCommandPoolCreateInfo cpci = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = g->qf,
	};
	CK(g->CreateCommandPool(g->dev, &cpci, nullptr, &g->cp));
	VkCommandBufferAllocateInfo cbai = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = g->cp,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	CK(g->AllocateCommandBuffers(g->dev, &cbai, &g->cb));
	VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	CK(g->CreateFence(g->dev, &fci, nullptr, &g->fence));
	return true;
}

/* Samples in the shader's layout (see fit_gpu.comp). */
static void fill_samples(void *ctx, void *dst, size_t off, size_t n)
{
	const sample *sm = ctx;
	uint32_t *o = dst;
	const size_t first = off / (GPU_SW * 4), count = n / (GPU_SW * 4);
	for (size_t i = 0; i < count; i++, o += GPU_SW) {
		const sample *s = &sm[first + i];
		for (unsigned q = 0; q < 12; q++)
			o[q] = (uint16_t)s->st[2 * q] |
			    (uint32_t)(uint16_t)s->st[2 * q + 1] << 16;
		o[12] = (uint16_t)s->st[24] | 256u << 16; /* bias input 1.0 */
		o[13] = s->c0 | (uint32_t)s->c1 << 8 | (uint32_t)s->msel << 16 |
		    (uint32_t)s->bit << 24;
	}
}

void taw_gpu_close(taw_gpu *g)
{
	if (!g)
		return;
	if (g->dev) {
		rmbuf(g, &g->smp);
		rmbuf(g, &g->stage);
		if (g->fence)
			g->DestroyFence(g->dev, g->fence, nullptr);
		if (g->cp)
			g->DestroyCommandPool(g->dev, g->cp, nullptr);
		if (g->dp)
			g->DestroyDescriptorPool(g->dev, g->dp, nullptr);
		if (g->pipe)
			g->DestroyPipeline(g->dev, g->pipe, nullptr);
		if (g->prep)
			g->DestroyPipeline(g->dev, g->prep, nullptr);
		if (g->pl)
			g->DestroyPipelineLayout(g->dev, g->pl, nullptr);
		if (g->dsl)
			g->DestroyDescriptorSetLayout(g->dev, g->dsl, nullptr);
		g->DestroyDevice(g->dev, nullptr);
	}
	if (g->inst)
		g->DestroyInstance(g->inst, nullptr);
	if (g->lib)
		dlclose(g->lib);
	free(g);
}

static_assert(GPU_SW * 4 * 1024 <= GPU_STAGE, "staging holds whole samples");

taw_status taw_gpu_open(const sample *sm, size_t ns, bool debug,
    taw_gpu **out)
{
	*out = nullptr;
	taw_gpu *g = calloc(1, sizeof *g);
	if (!g)
		return TAW_E_NOMEM;
	g->debug = debug;
	g->ns = ns;
	const uint64_t smp_bytes = (uint64_t)ns * GPU_SW * 4;
	if (ns == 0 || ns > (1u << 23) || smp_bytes > GPU_VRAM_MAX / 2) {
		note(g, "%zu samples are out of range", ns);
		goto fail;
	}
	g->lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
	if (!g->lib)
		g->lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
	if (!g->lib) {
		note(g, "no Vulkan loader (%s)", dlerror());
		goto fail;
	}
	void *sym = dlsym(g->lib, "vkGetInstanceProcAddr");
	if (!sym) {
		note(g, "no vkGetInstanceProcAddr");
		goto fail;
	}
	memcpy(&g->GetInstanceProcAddr, &sym, sizeof sym);
	g->CreateInstance = (PFN_vkCreateInstance)g->GetInstanceProcAddr(
	    VK_NULL_HANDLE, "vkCreateInstance");
	if (!g->CreateInstance || !pick_device(g) || !make_pipeline(g) ||
	    !mkbuf(g, GPU_STAGE, true, &g->stage) ||
	    !mkbuf(g, smp_bytes, false, &g->smp) ||
	    !xfer(g, g->smp.b, true, nullptr, (size_t)smp_bytes, fill_samples,
		(void *)sm))
		goto fail;
	if (debug)
		fprintf(stderr, "taw fit: gpu: %s, %zu samples (%.0f MiB)\n",
		    g->name, ns, (double)smp_bytes / 1048576.0);
	*out = g;
	return TAW_OK;
fail:
	taw_gpu_close(g);
	return TAW_E_GPU;
}

static void push(taw_gpu *g, const gpu_push *pc, uint32_t gx, uint32_t gy)
{
	g->CmdPushConstants(g->cb, g->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
	    sizeof *pc, pc);
	g->CmdDispatch(g->cb, gx, gy, 1);
	VkMemoryBarrier mb = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
		    VK_ACCESS_SHADER_WRITE_BIT,
	};
	g->CmdPipelineBarrier(g->cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0,
	    nullptr);
}

/* A net's layout in the parameter buffer, in vec4s (see fit_gpu.comp):
 * rows padded to whole vec4s. */
typedef struct {
	uint64_t q1, q2;  /* vec4s per layer-1 set, per output set */
	uint64_t n14, np4; /* layer 1, both layers */
	uint64_t size;    /* w, m, v, (b1^t, b2^t), partial-sum slots */
} gpu_layout;

static gpu_layout layout_of(const tnet *t)
{
	const unsigned H = t->s.hidden;
	gpu_layout l = {.q1 = H * 7u, .q2 = H ? (H + 4u) / 4u : 7u};
	l.n14 = t->s1 * l.q1;
	l.np4 = l.n14 + t->s2 * l.q2;
	l.size = 3 * l.np4 + 1 + 16 * (l.q1 + l.q2);
	return l;
}

/* Between tnet rows (row1 = H x P1_F per set, row2) and padded vec4 rows.
 * taw_gpu_train rejects t->f != P1_F before this is called. */
static void pad_rows(const tnet *t, const gpu_layout *l, float *gpu,
    float *cpu, bool to_gpu)
{
	const unsigned H = t->s.hidden;
	for (size_t set = 0; set < t->s1; set++)
		for (unsigned j = 0; j < H; j++) {
			float *gr = gpu + 4 * (set * l->q1 + j * 7u);
			float *cr = cpu + set * t->row1 + j * P1_F;
			if (to_gpu)
				memcpy(gr, cr, P1_F * sizeof(float));
			else
				memcpy(cr, gr, P1_F * sizeof(float));
		}
	for (size_t set = 0; set < t->s2; set++) {
		float *gr = gpu + 4 * (l->n14 + set * l->q2);
		float *cr = cpu + t->n1 + set * t->row2;
		if (to_gpu)
			memcpy(gr, cr, t->row2 * sizeof(float));
		else
			memcpy(cr, gr, t->row2 * sizeof(float));
	}
}

static bool train(taw_gpu *g, tnet *nets, size_t k, int epochs,
    const uint64_t *seeds, gbuf *pb, gbuf *mb, gbuf *bb, float **hp,
    uint32_t **hm)
{
	const uint64_t ns = g->ns, stride = fit_stride(ns);
	const uint64_t per_epoch = (ns + GPU_NB - 1) / GPU_NB;
	const uint64_t chunks = (per_epoch + GPU_CHUNK - 1) / GPU_CHUNK;
	const uint64_t steps_total = (uint64_t)epochs * per_epoch;
	if (steps_total > UINT32_MAX || k > 65535)
		return false;

	uint64_t total = 0; /* vec4s */
	for (size_t i = 0; i < k; i++)
		total += layout_of(&nets[i]).size;
	const uint64_t blk_bytes = (uint64_t)k * GPU_CHUNK * GPU_BLK * 4;
	if (total * 16 + blk_bytes + g->smp.size > GPU_VRAM_MAX ||
	    total > UINT32_MAX / 4) {
		note(g, "%llu MiB of parameters is too much",
		    (unsigned long long)(total >> 16));
		return false;
	}
	const size_t meta_n = GPU_DESC + 8 * k + (size_t)epochs * chunks * k;
	float *P = *hp = calloc((size_t)total * 4, sizeof *P);
	uint32_t *M = *hm = calloc(meta_n, sizeof *M);
	if (!P || !M)
		return false;
	for (int t = -2047; t <= 2047; t++)
		M[t + 2047] = p1_squash(t);
	uint64_t base = 0;
	for (size_t i = 0; i < k; i++) {
		const tnet *t = &nets[i];
		const gpu_layout l = layout_of(t);
		uint32_t *d = &M[GPU_DESC + 8 * i];
		d[0] = t->s.hidden;
		d[1] = t->s.sel1;
		d[2] = t->s.sel2;
		d[3] = (uint32_t)base;
		d[4] = (uint32_t)l.np4;
		d[5] = (uint32_t)l.n14;
		d[6] = (uint32_t)t->s1;
		d[7] = (uint32_t)t->s2;
		pad_rows(t, &l, P + 4 * base, t->w, true);
		P[4 * (base + 3 * l.np4)] = 1.0f;     /* b1^t */
		P[4 * (base + 3 * l.np4) + 1] = 1.0f; /* b2^t */
		base += l.size;
		/* each dispatch's first sample, as tnet_train draws them */
		uint64_t rs = seeds[i];
		for (int e = 0; e < epochs; e++) {
			const uint64_t p0 = rng_next(&rs) % ns;
			for (uint64_t c = 0; c < chunks; c++)
				M[GPU_DESC + 8 * k + ((size_t)e * chunks + c) * k + i] =
				    (uint32_t)((p0 + c * GPU_CHUNK * GPU_NB % ns * stride) % ns);
		}
	}
	if (!mkbuf(g, total * 16, false, pb) ||
	    !mkbuf(g, meta_n * 4, false, mb) ||
	    !mkbuf(g, blk_bytes, false, bb) ||
	    !xfer(g, pb->b, true, P, (size_t)total * 16, nullptr, nullptr) ||
	    !xfer(g, mb->b, true, M, meta_n * 4, nullptr, nullptr))
		return false;
	VkDescriptorBufferInfo dbi[4] = {
		{.buffer = g->smp.b, .range = VK_WHOLE_SIZE},
		{.buffer = pb->b, .range = VK_WHOLE_SIZE},
		{.buffer = mb->b, .range = VK_WHOLE_SIZE},
		{.buffer = bb->b, .range = VK_WHOLE_SIZE},
	};
	VkWriteDescriptorSet wr[4];
	for (uint32_t i = 0; i < 4; i++)
		wr[i] = (VkWriteDescriptorSet){
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = g->ds,
			.dstBinding = i,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &dbi[i],
		};
	g->UpdateDescriptorSets(g->dev, 4, wr, 0, nullptr);

	/* The whole fit in one command buffer. */
	if (!begin(g))
		return false;
	g->CmdBindDescriptorSets(g->cb, VK_PIPELINE_BIND_POINT_COMPUTE, g->pl, 0,
	    1, &g->ds, 0, nullptr);
	const int warm = fit_warm_epochs(epochs);
	gpu_push pc = {.steps_total = (uint32_t)steps_total,
		.ns = (uint32_t)ns, .stride = (uint32_t)stride,
		.step_nb = (uint32_t)(GPU_NB * stride % ns),
		.ncand = (uint32_t)k};
	uint64_t step = 0;
	for (int e = 0; e < epochs; e++) {
		if (e == warm) {
			pc.mode = 1;
			g->CmdBindPipeline(g->cb, VK_PIPELINE_BIND_POINT_COMPUTE,
			    g->pipe);
			push(g, &pc, (uint32_t)k, 1);
			pc.mode = 0;
		}
		pc.tied = e < warm;
		for (uint64_t c = 0; c < chunks; c++) {
			const uint64_t b0 = c * GPU_CHUNK;
			const uint64_t nb = per_epoch - b0 < GPU_CHUNK ? per_epoch - b0
								      : GPU_CHUNK;
			pc.nbatch = (uint32_t)nb;
			pc.last_nb = (uint32_t)((b0 + nb) * GPU_NB <= ns
				? GPU_NB : ns - (b0 + nb - 1) * GPU_NB);
			pc.step0 = (uint32_t)step;
			pc.pos_off = (uint32_t)(GPU_DESC + 8 * k +
			    ((size_t)e * chunks + c) * k);
			g->CmdBindPipeline(g->cb, VK_PIPELINE_BIND_POINT_COMPUTE,
			    g->prep);
			push(g, &pc, (uint32_t)nb, (uint32_t)k);
			g->CmdBindPipeline(g->cb, VK_PIPELINE_BIND_POINT_COMPUTE,
			    g->pipe);
			push(g, &pc, (uint32_t)k, 1);
			step += nb;
		}
	}
	if (!submit_wait(g) ||
	    !xfer(g, pb->b, false, P, (size_t)total * 16, nullptr, nullptr))
		return false;
	base = 0;
	for (size_t i = 0; i < k; i++) {
		const gpu_layout l = layout_of(&nets[i]);
		pad_rows(&nets[i], &l, P + 4 * base, nets[i].w, false);
		base += l.size;
	}
	return true;
}

taw_status taw_gpu_train(taw_gpu *g, tnet *nets, size_t k, int epochs,
    const uint64_t *seeds)
{
	if (k == 0)
		return TAW_OK;
	if (!nets)
		return TAW_E_ARG;
	for (size_t i = 0; i < k; i++)
		if (!gpu_net_fits(&nets[i]))
			return TAW_E_ARG;
	gbuf pb = {0}, mb = {0}, bb = {0};
	float *hp = nullptr;
	uint32_t *hm = nullptr;
	bool ok = train(g, nets, k, epochs, seeds, &pb, &mb, &bb, &hp, &hm);
	rmbuf(g, &pb);
	rmbuf(g, &mb);
	rmbuf(g, &bb);
	free(hp);
	free(hm);
	return ok ? TAW_OK : TAW_E_GPU;
}

#else /* no GPU support in this build */

taw_status taw_gpu_open(const sample *sm, size_t ns, bool debug,
    taw_gpu **out)
{
	(void)sm;
	(void)ns;
	if (debug)
		fprintf(stderr, "taw fit: gpu: not built (TADAMW_GPU=OFF)\n");
	*out = nullptr;
	return TAW_E_GPU;
}

taw_status taw_gpu_train(taw_gpu *g, tnet *nets, size_t k, int epochs,
    const uint64_t *seeds)
{
	(void)g;
	(void)epochs;
	(void)seeds;
	if (k && (!nets))
		return TAW_E_ARG;
	for (size_t i = 0; i < k; i++)
		if (!gpu_net_fits(&nets[i]))
			return TAW_E_ARG;
	return TAW_E_GPU;
}

void taw_gpu_close(taw_gpu *g)
{
	(void)g;
}

#endif
