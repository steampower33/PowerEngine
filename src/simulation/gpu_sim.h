#pragma once

class Context;
class Swapchain;
class Texture;
class TextureManager;
class Model;
class ModelManager;
class MouseInteractor;

#include "sim_data.h"
#include "sim_ubo.h"

#include "vulkan_utils.h"

#include <vk_radix_sort.h>

class GpuSim
{
public:
	GpuSim(
		Context& context,
		Swapchain& swapchain,
		TextureManager& textureManager,
		ModelManager& modelManager,
		vk::raii::DescriptorSetLayout& globalSetLayout,
		std::vector<vk::Format>& formats,
		vk::raii::DescriptorSetLayout& tex2DSetLayout);
	GpuSim(const GpuSim& rhs) = delete;
	GpuSim(GpuSim&& rhs) = delete;
	GpuSim& operator=(const GpuSim& rhs) = delete;
	GpuSim& operator=(GpuSim&& rhs) = delete;
	~GpuSim();

	void UpdateComputeUBO(uint32_t currentFrame, std::unique_ptr<Model>& model);
	void UpdateGraphicsUBO(uint32_t currentFrame);
	void UpdateMousePushConstant(Camera& camera, MouseInteractor& mouseInteractor, glm::vec2 viewportSize);

	void RecordCompute(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd, vk::raii::QueryPool& timestampPool, uint32_t& timestampSteps, vku::TestScene& testScene);
	void RecordGraphics(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd, vk::raii::DescriptorSet& globalSet, uint32_t globalOffset, vku::PolygonMode mode,
		vk::raii::DescriptorSet& tex2DSet);

	void CopyDatas(const vk::raii::CommandBuffer& cmd);
	void UpdateTestScene(const vk::raii::CommandBuffer& cmd, vku::TestScene& testScene);

	SimData datas_;
	SimUBO ubo_;

	uint32_t iteration_timestamp_count_ = 12;

	vku::Counts counts_;
	vk::raii::DescriptorPool descriptor_pool_{ nullptr };

	vk::raii::Buffer index_buffer_{ nullptr };
	vk::raii::DeviceMemory index_buffer_memory_{ nullptr };

	struct RadixSortContext
	{
		VrdxSorter sorter = VK_NULL_HANDLE;
		vk::raii::Buffer       storage_buffer{ nullptr };
		vk::raii::DeviceMemory storage_memory{ nullptr };
		vk::DeviceSize         storage_size = 0;
	} radix_;

	struct PushConstant {
		struct Solve {
			uint32_t base;
			uint32_t count;
			float compliance;
			float beta;
		} solve;
		static_assert(sizeof(Solve) % 4 == 0, "push constant must be multiple of 4 bytes");

		struct MouseInteract {
			glm::vec3 ray_origin;
			uint32_t select_mode; // 0: none, 1: select, 2: drag
			glm::vec3 ray_dir;
			float radius = 0.1f;
			uint32_t depth_mode;
		} mouse_interact;
		static_assert(sizeof(MouseInteract) % 4 == 0, "push constant must be multiple of 4 bytes");

		struct ClothRender {
			uint32_t nx1;
			uint32_t ny1;
		} cloth_render;
		static_assert(sizeof(ClothRender) % 4 == 0, "push constant must be multiple of 4 bytes");

	} push_constants_;

	struct SetLayout {
		vk::raii::DescriptorSetLayout sim_params{ nullptr };
		vk::raii::DescriptorSetLayout render{ nullptr };
		vk::raii::DescriptorSetLayout cloth_compute{ nullptr };
		vk::raii::DescriptorSetLayout cloth_graphics{ nullptr };
	} set_layouts_;

	struct Set {
		vk::raii::DescriptorSet sim_params{ nullptr };
		vk::raii::DescriptorSet render{ nullptr };
		vk::raii::DescriptorSet cloth_compute{ nullptr };
		vk::raii::DescriptorSet cloth_graphics{ nullptr };
	} sets_;

	struct PipelineLayout {
		vk::raii::PipelineLayout common{ nullptr };
		vk::raii::PipelineLayout cloth_graphics{ nullptr };
	} pipeline_layouts_;

	struct Pipeline {
		vk::raii::Pipeline clear_lambdas{ nullptr };
		vk::raii::Pipeline integrate{ nullptr };
		vk::raii::Pipeline solve_stretch{ nullptr };
		vk::raii::Pipeline solve_shear{ nullptr };
		vk::raii::Pipeline solve_bend{ nullptr };
		vk::raii::Pipeline solve_area{ nullptr };
		vk::raii::Pipeline apply_deltas{ nullptr };
		vk::raii::Pipeline collide_sdf{ nullptr };
		vk::raii::Pipeline update_velocity{ nullptr };

		vk::raii::Pipeline cloth_solid{ nullptr };
		vk::raii::Pipeline cloth_wireframe{ nullptr };
		vk::raii::Pipeline cloth_point{ nullptr };

		vk::raii::Pipeline build_hash{ nullptr };
		vk::raii::Pipeline build_cell{ nullptr };
		vk::raii::Pipeline build_neighbor{ nullptr };
		vk::raii::Pipeline solve_self_collision{ nullptr };

	} pipelines_;

	struct SSBO {
		vk::raii::Buffer position{ nullptr };
		vk::raii::Buffer pred_position{ nullptr };
		vk::raii::Buffer velocity{ nullptr };
		vk::raii::Buffer inverse_mass{ nullptr };
		vk::raii::Buffer delta_x{ nullptr };
		vk::raii::Buffer delta_y{ nullptr };
		vk::raii::Buffer delta_z{ nullptr };
		vk::raii::Buffer delta_count{ nullptr };
		vk::raii::Buffer edge{ nullptr };
		vk::raii::Buffer shear{ nullptr };
		vk::raii::Buffer bend{ nullptr };
		vk::raii::Buffer grab_state{ nullptr };
		vk::raii::Buffer area{ nullptr };

		vk::raii::Buffer particle_hash{ nullptr };
		vk::raii::Buffer particle_indice{ nullptr };
		vk::raii::Buffer start{ nullptr };
		vk::raii::Buffer end{ nullptr };
		vk::raii::Buffer neighbor{ nullptr };
		vk::raii::Buffer neighbor_lambda{ nullptr };
	} ssbos_;

	struct SSBOMemory {
		vk::raii::DeviceMemory position{ nullptr };
		vk::raii::DeviceMemory pred_position{ nullptr };
		vk::raii::DeviceMemory velocity{ nullptr };
		vk::raii::DeviceMemory inverse_mass{ nullptr };
		vk::raii::DeviceMemory delta_x{ nullptr };
		vk::raii::DeviceMemory delta_y{ nullptr };
		vk::raii::DeviceMemory delta_z{ nullptr };
		vk::raii::DeviceMemory delta_count{ nullptr };
		vk::raii::DeviceMemory edge{ nullptr };
		vk::raii::DeviceMemory shear{ nullptr };
		vk::raii::DeviceMemory bend{ nullptr };
		vk::raii::DeviceMemory grab_state{ nullptr };
		vk::raii::DeviceMemory area{ nullptr };

		vk::raii::DeviceMemory particle_hash{ nullptr };
		vk::raii::DeviceMemory particle_indice{ nullptr };
		vk::raii::DeviceMemory start{ nullptr };
		vk::raii::DeviceMemory end{ nullptr };
		vk::raii::DeviceMemory neighbor{ nullptr };
		vk::raii::DeviceMemory neighbor_lambda{ nullptr };
	} ssbo_memories_;

	struct SSBOSize {
		vk::DeviceSize position = 0;
		uint32_t pred_position = 0;
		uint32_t velocity = 0;
		uint32_t inverse_mass = 0;
		uint32_t delta_x = 0;
		uint32_t delta_y = 0;
		uint32_t delta_z = 0;
		uint32_t delta_count = 0;
		uint32_t edge = 0;
		uint32_t shear = 0;
		uint32_t bend = 0;
		uint32_t grab_state = 0;
		uint32_t area = 0;

		uint32_t particle_hash = 0;
		uint32_t particle_indice = 0;
		uint32_t start = 0;
		uint32_t end = 0;
		uint32_t neighbor = 0;
		uint32_t neighbor_lambda = 0;
	} ssbo_size_;

	struct Staging {
		vk::raii::Buffer position{ nullptr };
		vk::raii::Buffer pred_position{ nullptr };
		vk::raii::Buffer velocity{ nullptr };
		vk::raii::Buffer inverse_mass{ nullptr };
		vk::raii::Buffer edge{ nullptr };
		vk::raii::Buffer shear{ nullptr };
		vk::raii::Buffer bend{ nullptr };
		vk::raii::Buffer area{ nullptr };
	} staging_;

	struct StagingMemory {
		vk::raii::DeviceMemory position{ nullptr };
		vk::raii::DeviceMemory pred_position{ nullptr };
		vk::raii::DeviceMemory velocity{ nullptr };
		vk::raii::DeviceMemory inverse_mass{ nullptr };
		vk::raii::DeviceMemory edge{ nullptr };
		vk::raii::DeviceMemory shear{ nullptr };
		vk::raii::DeviceMemory bend{ nullptr };
		vk::raii::DeviceMemory area{ nullptr };
	} staging_memories_;

	struct StagingMapped {
		void* position{ nullptr };
		void* pred_position{ nullptr };
		void* velocity{ nullptr };
		void* inverse_mass{ nullptr };
		void* edge{ nullptr };
		void* shear{ nullptr };
		void* bend{ nullptr };
		void* area{ nullptr };
	} staging_mapped_;


private:
	Context& context_;
	Swapchain& swapchain_;
	TextureManager& texture_manager_;
	ModelManager& model_manager_;

private:
	void CreateDescriptorSetLayout();
	void CreateDescriptorPools();
	void CreateUniformBuffers();
	void CreateSSBOBuffers();
	void CreateDescriptorSets();
	void CreateComputePipelines();
	void CreateGraphicsPipelines(vk::raii::DescriptorSetLayout& globalSetLayout,
		std::vector<vk::Format>& formats,
		vk::raii::DescriptorSetLayout& tex2DSetLayout);
	void CreateVrdxSorter();
};