#pragma once

class Context;
class Swapchain;
class Texture;
class TextureManager;
class Model;
class ModelManager;
class MouseInteractor;
class ClothParticleManager;
class ClothGpuProfiler;

#include "cloth_sim_data.h"
#include "cloth_ubo.h"

#include "vulkan_utils.h"

#include <vk_radix_sort.h>

class ClothSimPass
{
public:
	ClothSimPass(Context& context, Swapchain& swapchain, ClothParticleManager& particleManager, ModelManager& modelManager);
	ClothSimPass(const ClothSimPass& rhs) = delete;
	ClothSimPass(ClothSimPass&& rhs) = delete;
	ClothSimPass& operator=(const ClothSimPass& rhs) = delete;
	ClothSimPass& operator=(ClothSimPass&& rhs) = delete;
	~ClothSimPass();

	Context& context_;
	ClothParticleManager& cloth_particle_manager_;
	ModelManager& model_manager_;

	std::unique_ptr<ClothGpuProfiler> gpu_profiler_;

	ClothData datas_;
	ClothUBO ubo_;

	std::vector<vk::raii::CommandBuffer> cmds_;

	void UpdateMousePushConstant(Camera& camera, MouseInteractor& mouseInteractor, glm::vec2 viewportSize);
	void UpdateComputeUBO(uint32_t currentFrame, ModelManager& model);

	void RecordCompute(uint32_t currentFrame);
	void CalculateGpuTime();

	bool is_reset_ = false;

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
		vk::raii::Pipeline solve_shear{ nullptr };
		vk::raii::Pipeline solve_bend{ nullptr };
		vk::raii::Pipeline solve_area{ nullptr };
		vk::raii::Pipeline solve_self_collision{ nullptr };
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
	void CreateColiiders();
	void CreateSSBOBuffers();
	void CreateDescriptorSets();
	void CreateComputePipelineLayouts();
	void CreateComputePipelines();
	void CreateVrdxSorter();

	void CopySimDatas(const vk::raii::CommandBuffer& cmd);
	void CopyColliders(const vk::raii::CommandBuffer& cmd);
};