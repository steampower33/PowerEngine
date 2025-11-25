#pragma once

#include "vulkan_utils.h"

class Context;
class Swapchain;
class CpuSim;
class GpuSim;
class Camera;
class Texture;
class TextureManager;
class Model;
class ModelManager;
class GUI;
class MouseInteractor;

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

	void Update(Camera& camera, MouseInteractor& mouseInteractor);
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

	std::array<std::string, 10> labels_ = { "Intergrate", "Clear Lambdas", "SolveStretch", "SolveShear", "SolveBend", "SolveArea", "ApplyDeltas", "CollideSdf", "Update", "Total" };
	std::unordered_map<std::string, double> label_time_;
	std::unordered_map<std::string, double> label_avg_time_;
	uint32_t time_count_ = 0;

	enum CpuOrGpu {
		CPU,
		GPU
	};

	CpuOrGpu cpu_or_gpu_ = CpuOrGpu::CPU;
	vku::TestScene test_scene_;

	std::unique_ptr<CpuSim> cpu_sim_;
	std::unique_ptr<GpuSim> gpu_sim_;

	vku::Counts counts_;

	struct UBO {
		vk::raii::Buffer global{ nullptr };
		vk::raii::Buffer object{ nullptr };
		vk::raii::Buffer light{ nullptr };
		vk::raii::Buffer skybox{ nullptr };
	} ubos_;

	struct UBOMemory {
		vk::raii::DeviceMemory global{ nullptr };
		vk::raii::DeviceMemory object{ nullptr };
		vk::raii::DeviceMemory light{ nullptr };
		vk::raii::DeviceMemory skybox{ nullptr };
	} ubo_memories_;

	struct UBOMapped {
		void* global{ nullptr };
		void* object{ nullptr };
		void* light{ nullptr };
		void* skybox{ nullptr };
	} ubo_mapped_;

	struct UBOSize {
		vk::DeviceSize global;
		vk::DeviceSize object;
		vk::DeviceSize light;
		vk::DeviceSize skybox;
	} ubo_size_;

	struct UBOData {
		struct Global {
			glm::mat4 view;
			glm::mat4 proj;

			uint32_t vulkan_thumbnail_index;
			uint32_t p1;
			uint32_t p2;
			uint32_t p3;
		} global;
		static_assert(sizeof(UBOData::Global) % 16 == 0, "std140 must be 16-byte aligned.");

		struct Object {
			glm::mat4 model;

			glm::vec4 albedo_use;

			int albedo_idx = -1;
			int metallic_idx = -1;
			int normal_idx = -1;
			int roughness_idx = -1;

			int ao_idx = -1;
			int height_idx = -1;
			float metallic_factor = 0.5f;
			float roughness_factor = 0.5f;

			float ao_factor = 0.0f;
			float height_factor = 0.0f;
			uint32_t p0 = 0;
			uint32_t p1 = 0;

			uint32_t albedo_enable = 0;
			uint32_t metallic_enable = 0;
			uint32_t normal_enable = 0;
			uint32_t roughness_enable = 0;

			uint32_t ao_enable = 0;
			uint32_t height_enable = 0;
			uint32_t checker_board_enable = 1;
			uint32_t p4;
		} object;
		static_assert(sizeof(UBOData::Object) % 16 == 0, "std140 must be 16-byte aligned.");

		struct Light {
			glm::mat4 invViewProj{};
			glm::vec4 cameraPos{};
			glm::vec3 position{ 0.0f, 5.0f, 5.0f };
			float intensity = 20.0f;
			glm::vec3 direction{ 0.0f, -1.0f, -1.0f };
			float inner = 0.0f;
			float outer = 90.0f;
			uint32_t light_enable = 1;
			uint32_t pbr_enable = 0;
			float exposure = 0.5f;
		} light;
		static_assert(sizeof(UBOData::Light) % 16 == 0, "std140 must be 16-byte aligned.");

		struct SkyBox {
			glm::mat4 model;

			int envIdx = 0;
			int radianceIdx = 0;
			int irradianceIdx = 0;
			uint32_t specularMipLevels = 0;

			int brdfLUTIndex = 0;
			uint32_t p0 = 0;
			uint32_t p1 = 0;
			uint32_t p2 = 0;
		} skybox;
		static_assert(sizeof(UBOData::SkyBox) % 16 == 0, "std140 must be 16-byte aligned.");

	} ubo_datas_;

	struct CommandBuffer {
		std::vector<vk::raii::CommandBuffer> compute;
		std::vector<vk::raii::CommandBuffer> graphics;
	} cmds_;

	struct SetLayout {
		vk::raii::DescriptorSetLayout global{ nullptr };
		vk::raii::DescriptorSetLayout object{ nullptr };
		vk::raii::DescriptorSetLayout tex2D{ nullptr };
		vk::raii::DescriptorSetLayout texEnv{ nullptr };
		vk::raii::DescriptorSetLayout lighting{ nullptr };
		vk::raii::DescriptorSetLayout skybox{ nullptr };
	} set_layouts_;

	struct Set {
		vk::raii::DescriptorSet global{ nullptr };
		vk::raii::DescriptorSet object{ nullptr };
		vk::raii::DescriptorSet tex2D{ nullptr };
		vk::raii::DescriptorSet texEnv{ nullptr };
		vk::raii::DescriptorSet lighting{ nullptr };
		vk::raii::DescriptorSet skybox{ nullptr };
	} sets_;

	struct PipelineLayout {
		vk::raii::PipelineLayout model{ nullptr };
		vk::raii::PipelineLayout lighting{ nullptr };
		vk::raii::PipelineLayout skybox{ nullptr };
	} pipeline_layouts_;

	struct Pipeline {
		vk::raii::Pipeline model_solid{ nullptr };
		vk::raii::Pipeline model_wireframe{ nullptr };
		vk::raii::Pipeline model_point{ nullptr };
		vk::raii::Pipeline lighting{ nullptr };
		vk::raii::Pipeline skybox{ nullptr };
	} pipelines_;

	vku::PolygonMode polygon_mode_ = vku::PolygonMode::SOLID;

	struct GeometryBuffer {
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

	vk::raii::Image depth_image_ = nullptr;
	vk::raii::DeviceMemory depth_image_memory_ = nullptr;
	vk::raii::ImageView depth_image_view_ = nullptr;

private:
	void RecreateSwapchain();
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