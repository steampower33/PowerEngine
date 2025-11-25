#pragma once

class Context;
class Swapchain;
class ModelManager;
class Texture;
class TextureManager;
class Ray;
class MouseInteractor;

#include "sim_data.h"
#include "sim_ubo.h"

class CpuSim {

public:
	CpuSim(
		Context& context,
		Swapchain& swapchain,
		TextureManager& textureManager,
		ModelManager& modelManager,
		vk::raii::DescriptorSetLayout& globalSetLayout,
		std::vector<vk::Format>& formats,
		vk::raii::DescriptorSetLayout& tex2DSetLayout);
	CpuSim(const CpuSim& rhs) = delete;
	CpuSim(CpuSim&& rhs) = delete;
	CpuSim& operator=(const CpuSim& rhs) = delete;
	CpuSim& operator=(CpuSim&& rhs) = delete;
	~CpuSim() = default;

	void UpdateMousePushConstant(Camera& camera, MouseInteractor& mouseInteractor, glm::vec2 viewportSize);
	void ComputeSolve(const glm::vec3& sphereCenter, float sphereRadius);
	void CopyPositions(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd);
	void UpdateGraphicsUBO(uint32_t currentFrame);
	void RecordGraphics(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd, vk::raii::DescriptorSet& globalSet, uint32_t globalOffset, vku::PolygonMode mode,
		vk::raii::DescriptorSet& tex2DSet);

	SimData datas_;
	SimUBO ubo_;

	int id = -1;
	float dist2 = 1000.0f;
	float T = 1000.0f;

	struct PushConstant {
		struct MouseInteract {
			glm::vec3 ray_origin;
			uint32_t select_mode; // 0: none, 1: select, 2: drag
			glm::vec3 ray_dir;
			float radius = 0.1f;
		} mouse_interact;
		static_assert(sizeof(MouseInteract) % 4 == 0, "push constant must be multiple of 4 bytes");

		struct ClothRender {
			uint32_t nx1;
			uint32_t ny1;
		} cloth_render;
		static_assert(sizeof(ClothRender) % 4 == 0, "push constant must be multiple of 4 bytes");

	} push_constants_;

	vku::Counts counts_;
	vk::raii::DescriptorPool descriptor_pool_{ nullptr };

	vk::raii::Buffer pos_ssbo_{ nullptr };
	vk::raii::DeviceMemory pos_ssbo_mem_{ nullptr };
	vk::raii::Buffer pos_staging_{ nullptr };
	vk::raii::DeviceMemory pos_staging_mem_{ nullptr };
	void* pos_staging_map_{ nullptr };
	vk::DeviceSize pos_ssbo_size_ = 0;

	vk::raii::Buffer index_buffer_{ nullptr };
	vk::raii::DeviceMemory index_buffer_memory_{ nullptr };

	struct SetLayout {
		vk::raii::DescriptorSetLayout render{ nullptr };
		vk::raii::DescriptorSetLayout cloth_graphics{ nullptr };
	} set_layouts_;

	struct Set {
		vk::raii::DescriptorSet render{ nullptr };
		vk::raii::DescriptorSet cloth_graphics{ nullptr };
	} sets_;

	struct PipelineLayout {
		vk::raii::PipelineLayout cloth_graphics{ nullptr };
	} pipeline_layouts_;

	struct Pipeline {
		vk::raii::Pipeline cloth_solid{ nullptr };
		vk::raii::Pipeline cloth_wireframe{ nullptr };
		vk::raii::Pipeline cloth_point{ nullptr };
	} pipelines_;

private:
	void CreateDescriptorSetLayout(Context& context);
	void CreateDescriptorPools(Context& context);
	void CreateUniformBuffers(Context& context,
		ModelManager& modelManager);
	void CreateDatas(Context& context);
	void CreateDescriptorSets(Context& context, TextureManager& textureManager);
	void CreateComputePipelines(Context& context);
	void CreateGraphicsPipelines(Context& context, vk::raii::DescriptorSetLayout& globalSetLayout,
		std::vector<vk::Format>& formats,
		vk::raii::DescriptorSetLayout& tex2DSetLayout);
};