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

	uint32_t frame_counter_ = 0;
	bool first_frame_ = (frame_counter_ == 0);

	vk::SampleCountFlagBits			 msaa_samples_ = vk::SampleCountFlagBits::e1;

	vk::raii::DescriptorPool		 descriptor_pool_{ nullptr };

	vk::raii::QueryPool				 timestamp_pool_{ nullptr };
	uint32_t timestamp_steps_ = 0;

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
	const float spacing_ = 0.05;

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

	struct UBO {
		vk::raii::Buffer global{ nullptr };
		vk::raii::Buffer object{ nullptr };
		vk::raii::Buffer light{ nullptr };
	} ubo_;

	struct UBOMem {
		vk::raii::DeviceMemory global{ nullptr };
		vk::raii::DeviceMemory object{ nullptr };
		vk::raii::DeviceMemory light{ nullptr };
	} ubo_memory_;

	struct UBOMapped {
		void* global{ nullptr };
		void* object{ nullptr };
		void* light{ nullptr };
	} ubo_mapped_;

	struct UBOSize {
		vk::DeviceSize global;
		vk::DeviceSize object;
		vk::DeviceSize light;
	} ubo_size_;

	struct UBOData {
		struct Global {
			glm::mat4 view;
			glm::mat4 proj;
		} global;

		struct Object {
			glm::mat4 model;
			glm::vec4 color_use;
			uint32_t albedo = 0;
			uint32_t ao = 0;
			uint32_t roughness = 0;
			uint32_t metallic = 0;
			uint32_t height = 0;
			uint32_t normal = 0;
			uint32_t chooseTexIdx = 0;
			uint32_t pad1 = 0;
		} object;

		struct Light {
			glm::vec4 cameraPos{};
			glm::vec4 spotPos_range{0.0f, 10.0f, 0.0f, 30.0f}; // xyz: 위치(월드), w: range(최대 거리)
			glm::vec4 spotDir_inner{0.0f, -1.0f, 0.0f, 0.0f}; // xyz: 방향(월드, normalized), w: innerConeCos
			glm::vec4 spotColor_outer{1.0f, 1.0f, 1.0f, 0.0f}; // rgb: color, w: outerConeCos
			glm::mat4 invViewProj{};
		} light;

	} ubo_data_;

	struct CommandBuffers {
		std::vector<vk::raii::CommandBuffer> compute;
		std::vector<vk::raii::CommandBuffer> graphics;
	} cmds_;

	struct SetLayouts {
		vk::raii::DescriptorSetLayout global{ nullptr };
		vk::raii::DescriptorSetLayout object{ nullptr };
		vk::raii::DescriptorSetLayout lighting{ nullptr };
	} set_layouts_;

	struct Sets {
		vk::raii::DescriptorSet global{ nullptr };
		vk::raii::DescriptorSet object{ nullptr };
		vk::raii::DescriptorSet lighting{ nullptr };
	} sets_;

	struct PipelineLayouts {
		vk::raii::PipelineLayout model{ nullptr };
		vk::raii::PipelineLayout lighting{ nullptr };
	} pipeline_layouts_;

	struct Pipelines {
		vk::raii::Pipeline model{ nullptr };
		vk::raii::Pipeline lighting{ nullptr };
	} pipelines_;

	struct GeometryBuffers {
		std::vector<vk::Format> formats;

		vk::raii::Sampler sampler{ nullptr };

		vk::raii::Image albedo_mettalic_image = nullptr;
		vk::raii::DeviceMemory albedo_mettalic_image_memory = nullptr;
		vk::raii::ImageView albedo_mettalic_image_view = nullptr;

		vk::raii::Image normal_roughness_image = nullptr;
		vk::raii::DeviceMemory normal_roughness_image_memory = nullptr;
		vk::raii::ImageView normal_roughness_image_view = nullptr;

		vk::raii::Image height_ao_image = nullptr;
		vk::raii::DeviceMemory height_ao_image_memory = nullptr;
		vk::raii::ImageView height_ao_image_view = nullptr;

	} geometry_buffers_;

	glm::vec3 background_color_ = glm::vec3(glm::pow(214.0f / 255.0f, 2.2f), glm::pow(225.0f / 255.0f, 2.2f), glm::pow(252.0f / 255.0f, 2.2f));

	// |===== Depth Image =====|
	vk::raii::Image depth_image_ = nullptr;
	vk::raii::DeviceMemory depth_image_memory_ = nullptr;
	vk::raii::ImageView depth_image_view_ = nullptr;

private:
	void RecreateSwapchain();
	void UpdatePushContants();
	void UpdateGraphicsUBO(Camera& camera);

	void RecordGraphicsCommandBuffer(uint32_t imageIndex);

private:
	void CreateCommandBuffers();
	void CreateQueryPool();

	void CreateDescriptorSetLayout();
	void CreateDescriptorPools();

	void CreateUniformBuffers();

	void CreateDescriptorSets();
	void CreateGraphicsPipelines();
	void CreateSyncObjects();

	void CreateGeometryBuffers();
	void CreateDepthResources();

};