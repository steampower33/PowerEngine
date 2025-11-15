#pragma once

#include "vulkan_utils.h"

class Context;
class Swapchain;
class CpuSim;
class GpuSim;
class Camera;
class Texture2D;
class TextureManager;
class Model;
class ModelManager;
class GUI;

class GraphicsContext
{
public:
	GraphicsContext(GLFWwindow* glfwWindow, Context& context, Swapchain& swapchain, TextureManager& textureManager, ModelManager& modelManager);
	GraphicsContext(const GraphicsContext& rhs) = delete;
	GraphicsContext(GraphicsContext&& rhs) = delete;
	GraphicsContext& operator=(const GraphicsContext& rhs) = delete;
	GraphicsContext& operator=(GraphicsContext&& rhs) = delete;
	~GraphicsContext();

	Context& context_;
	Swapchain& swapchain_;
	TextureManager& texture_manager_;
	ModelManager& model_manager_;

	void Update(Camera& camera);
	void Draw(std::unique_ptr<GUI>& gui);

	vk::SampleCountFlagBits			 msaa_samples_ = vk::SampleCountFlagBits::e1;

	vk::raii::DescriptorPool		 descriptor_pool_{ nullptr };

	vk::raii::QueryPool				 timestamp_pool_{ nullptr };
	uint32_t timestampSteps = 0;

	vk::raii::Semaphore semaphore_{ nullptr };
	uint64_t timeline_value_{ 0 };
	std::vector<vk::raii::Fence> in_flight_fences_;
	uint32_t current_frame_{ 0 };
	uint32_t read_set_{ 0 };

	std::array<std::string, 9> labels_ = { "Intergrate", "Clear Lambdas", "SolveStretch", "SolveDiagonal", "SolveBend", "ApplyDeltas", "CollideSdf", "Update", "Total" };
	std::unordered_map<std::string, double> label_time_;
	std::unordered_map<std::string, double> label_avg_time_;
	uint32_t time_count_ = 0;

	enum CpuOrGpu {
		CPU,
		GPU
	};

	CpuOrGpu cpu_or_gpu_ = CpuOrGpu::GPU;
	vku::TestScene test_scene_;

	std::unique_ptr<CpuSim> cpu_sim_;
	std::unique_ptr<GpuSim> gpu_sim_;

	// |===== Particle Info =====|
	const uint32_t Nx_ = 64;
	const uint32_t Ny_ = 64;
	const float spacing_ = 0.1;

	uint32_t particles_size_ = Nx_ * Ny_;
	uint32_t indices_size_ = 0;
	uint32_t edges_size_ = 0;

	// |===== Push Constant =====|
	struct ClothPC {
		uint32_t Nx;
		uint32_t Ny;
	} cloth_pc_;
	static_assert(sizeof(ClothPC) % 4 == 0, "push constant must be multiple of 4 bytes");

	vku::Counts counts_;
	uint32_t sim_count = 0;

	struct CommandBuffers {
		std::vector<vk::raii::CommandBuffer> compute;
		std::vector<vk::raii::CommandBuffer> graphics;
	} cmds_;

	// |===== Graphics Info =====|
	struct Graphics {
		struct GlobalUboData {
			glm::mat4 view;
			glm::mat4 proj;
		} global_ubo_data;
		vk::raii::Buffer global_ubo{ nullptr };
		vk::raii::DeviceMemory global_ubo_memory{ nullptr };
		void* global_ubo_mapped{ nullptr };
		vk::DeviceSize global_slot_size;

		struct ObjectUboData {
			glm::mat4 model;
			glm::vec4 color_use;
		} object_ubo_data;
		vk::raii::Buffer object_ubo{ nullptr };
		vk::raii::DeviceMemory object_ubo_memory{ nullptr };
		void* object_ubo_mapped{ nullptr };
		vk::DeviceSize object_slot_size;

		vk::raii::DescriptorSetLayout global_set_layout{ nullptr };
		vk::raii::DescriptorSet global_set{ nullptr };
		vk::raii::DescriptorSetLayout object_set_layout{ nullptr };
		vk::raii::DescriptorSet object_set{ nullptr };

		struct PipelineLayouts {
			vk::raii::PipelineLayout model{ nullptr };
		} pipeline_layouts;

		struct Pipelines {
			vk::raii::Pipeline model{ nullptr };
		} pipelines;

	} graphics_;

	glm::vec3 background_color = glm::vec3(glm::pow(214.0f / 255.0f, 2.2f), glm::pow(225.0f / 255.0f, 2.2f), glm::pow(252.0f / 255.0f, 2.2f));

	// |===== Depth Image =====|
	vk::raii::Image depth_image_ = nullptr;
	vk::raii::DeviceMemory depth_image_memory_ = nullptr;
	vk::raii::ImageView depth_image_view_ = nullptr;

private:
	void RecreateSwapchain();
	void UpdatePushContants();
	void UpdateGraphicsUBO(Camera& camera);

	void RecordGraphicsCommandBuffer(uint32_t imageIndex);
	void TransitionImageLayout(
		vk::Image& image,
		const vk::raii::CommandBuffer& cmd,
		vk::ImageLayout old_layout,
		vk::ImageLayout new_layout,
		vk::AccessFlags2 src_access_mask,
		vk::AccessFlags2 dst_access_mask,
		vk::PipelineStageFlags2 src_stage_mask,
		vk::PipelineStageFlags2 dst_stage_mask
	);
	void TransitionImageLayoutCustom(
		vk::raii::Image& image,
		const vk::raii::CommandBuffer& cmd,
		vk::ImageLayout old_layout,
		vk::ImageLayout new_layout,
		vk::AccessFlags2 src_access_mask,
		vk::AccessFlags2 dst_access_mask,
		vk::PipelineStageFlags2 src_stage_mask,
		vk::PipelineStageFlags2 dst_stage_mask,
		vk::ImageAspectFlags aspect_mask
	);

private:
	vk::SampleCountFlagBits GetMaxUsableSampleCount();

	void CreateCommandBuffers();
	void CreateQueryPool();

	void CreateDescriptorSetLayout();
	void CreateDescriptorPools();

	void CreateUniformBuffers();

	void CreateDescriptorSets();
	void CreateGraphicsPipelines();
	void CreateSyncObjects();

	void CreateDepthResources();

};