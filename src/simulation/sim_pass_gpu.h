#pragma once

class Context;
class Swapchain;
class Texture;
class TextureManager;
class Model;
class ModelManager;
class MouseInteractor;
class ParticleManager;
class GpuProfiler;
struct SimData;
struct SimUBO;

#include "sim_data.h"
#include "sim_ubo.h"

#include "vulkan_utils.h"

#include <vk_radix_sort.h>

class SimPassGPU
{
public:
	SimPassGPU(Context& context, Swapchain& swapchain, ParticleManager& particleManager, ModelManager& modelManager);
	SimPassGPU(const SimPassGPU& rhs) = delete;
	SimPassGPU(SimPassGPU&& rhs) = delete;
	SimPassGPU& operator=(const SimPassGPU& rhs) = delete;
	SimPassGPU& operator=(SimPassGPU&& rhs) = delete;
	~SimPassGPU();

	Context& context_;
	ParticleManager& particle_manager_;
	ModelManager& model_manager_;

	std::unique_ptr<GpuProfiler> gpu_profiler_;

	SimData sim_datas_;
	SimUBO ubo_;

	std::vector<vk::raii::CommandBuffer> cmds_;

	void UpdateMousePushConstant(Camera& camera, MouseInteractor& mouseInteractor, glm::vec2 viewportSize);
	void UpdateComputeUBO(uint32_t currentFrame, ModelManager& model);

	void RecordCompute(uint32_t currentFrame, vku::TestScene& testScene);
	void CalculateGpuTime();

	vku::Count counts_;
	vk::raii::DescriptorPool descriptor_pool_{ nullptr };

	struct RadixSortContext
	{
		VrdxSorter sorter = VK_NULL_HANDLE;
		vk::raii::Buffer       storage_buffer{ nullptr };
		vk::raii::DeviceMemory storage_memory{ nullptr };
		vk::DeviceSize         storage_size = 0;
	} radix_;

	struct SetLayout {
		vk::raii::DescriptorSetLayout sim_params{ nullptr };
		vk::raii::DescriptorSetLayout cloth_compute{ nullptr };
	} set_layouts_;

	struct Set {
		vk::raii::DescriptorSet sim_params{ nullptr };
		vk::raii::DescriptorSet cloth_compute{ nullptr };
	} sets_;

	struct PipelineLayout {
		vk::raii::PipelineLayout common{ nullptr };
	} pipeline_layouts_;

	struct Pipeline {
		vk::raii::Pipeline wind{ nullptr };
		vk::raii::Pipeline integrate{ nullptr };
		vk::raii::Pipeline clear_lambdas{ nullptr };
		vk::raii::Pipeline solve_stretch{ nullptr };
		vk::raii::Pipeline solve_softbody_stretch{ nullptr };
		vk::raii::Pipeline solve_softbody_volume{ nullptr };
		vk::raii::Pipeline solve_shear{ nullptr };
		vk::raii::Pipeline solve_bend{ nullptr };
		vk::raii::Pipeline solve_area{ nullptr };
		vk::raii::Pipeline solve_self_collision{ nullptr };
		vk::raii::Pipeline solve_inter_cloth_collision{ nullptr };
		vk::raii::Pipeline apply_deltas{ nullptr };
		vk::raii::Pipeline solve_lra{ nullptr };
		vk::raii::Pipeline collide_sdf{ nullptr };
		vk::raii::Pipeline update_velocity{ nullptr };

		vk::raii::Pipeline build_hash{ nullptr };
		vk::raii::Pipeline build_cell{ nullptr };
		vk::raii::Pipeline build_neighbor{ nullptr };

		vk::raii::Pipeline tri_normal{ nullptr };
		vk::raii::Pipeline vector_normal{ nullptr };

	} pipelines_;

private:
	void CreateCommandBuffers();
	void CreateDescriptorSetLayout();
	void CreateDescriptorPools();
	void CreateUniformBuffers();
	void CreateClothConstraintDatas();
	void CreateSoftBodyConstraintDatas();
	void CreateColiiders();
	void CreateSSBOBuffers();
	void CreateDescriptorSets();
	void CreateComputePipelineLayouts();
	void CreateComputePipelines();
	void CreateVrdxSorter();

	void CopySimDatas(const vk::raii::CommandBuffer& cmd);
	void ResetTestScene(const vk::raii::CommandBuffer& cmd, vku::TestScene& testScene);
	void CopyColliders(const vk::raii::CommandBuffer& cmd);
};