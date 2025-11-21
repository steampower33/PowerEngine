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

			uint32_t vulkanThumbnailIndex;
			uint32_t p1;
			uint32_t p2;
			uint32_t p3;
		} global;
		static_assert(sizeof(UBOData::Global) % 16 == 0, "std140 must be 16-byte aligned.");

		struct Object {
			glm::mat4 model;

			glm::vec4 albedo_use;

			int albedoIdx = -1;
			int metallicIdx = -1;
			int normalIdx = -1;
			int roughnessIdx = -1;

			int aoIdx = -1;
			int heightIdx = -1;
			float metallicFactor = 0.5f;
			float roughnessFactor = 0.5f;

			float aoFactor = 0.0f;
			float heightFactor = 0.0f;
			uint32_t p0 = 0;
			uint32_t p1 = 0;

			uint32_t albedoEnable = 0;
			uint32_t metallicEnable = 0;
			uint32_t normalEnable = 0;
			uint32_t roughnessEnable = 0;

			uint32_t aoEnable = 0;
			uint32_t heightEnable = 0;
			uint32_t p3;
			uint32_t p4;
		} object;
		static_assert(sizeof(UBOData::Object) % 16 == 0, "std140 must be 16-byte aligned.");

		struct Light {
			glm::vec4 cameraPos{};
			glm::vec4 spotPos_range{ 0.0f, 10.0f, 0.0f, 30.0f };
			glm::vec4 spotDir_inner{ 0.0f, -1.0f, 0.0f, 0.0f };
			glm::vec4 spotColor_outer{ 1.0f, 1.0f, 1.0f, 0.0f };
			glm::mat4 invViewProj{};
			float exposure = 0.5f;
			float p0;
			float p1;
			float p2;
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

	} ubo_data_;

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