#pragma once

class Context;
class Camera;
class Swapchain;
class Texture;
class TextureManager;
class Model;
class ModelManager;
class ParticleManager;

class GraphicsPass
{
public:
	GraphicsPass(Context& context, Swapchain& swapchain, TextureManager& textureManager, ModelManager& modelManager, ParticleManager& particleManager);
	GraphicsPass(const GraphicsPass& rhs) = delete;
	GraphicsPass(GraphicsPass&& rhs) = delete;
	GraphicsPass& operator=(const GraphicsPass& rhs) = delete;
	GraphicsPass& operator=(GraphicsPass&& rhs) = delete;
	~GraphicsPass();

	void UpdateGraphicsUBO(uint32_t currentFrame, Camera& camera);
	void RecordGraphicsCommandBuffer(uint32_t imageIndex, uint32_t currentFrame, vku::CpuOrGpu cpuOrGpu);
	void CreateDepthResources();
	
	void CalculateGpuTime();

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
			float metallic_factor = 0.0f;
			float roughness_factor = 0.0f;

			float ao_factor = 0.0f;
			float height_factor = 0.0f;
			float sheen_weight_factor = 0.0f;
			float sheen_roughness_factor = 0.0f;

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
			uint32_t light_enable = 0;
			uint32_t pbr_enable = 1;
			float exposure = 0.8f;
		} light;
		static_assert(sizeof(UBOData::Light) % 16 == 0, "std140 must be 16-byte aligned.");

		struct SkyBox {
			int env_idx = 0;
			int specular_idx = 0;
			int diffuse_idx = 0;
			uint32_t specular_mip_levels = 0;

			int brdfIndex = 0;
			uint32_t p0 = 0;
			uint32_t p1 = 0;
			uint32_t p2 = 0;
		} skybox;
		static_assert(sizeof(UBOData::SkyBox) % 16 == 0, "std140 must be 16-byte aligned.");

		struct Cloth {
			glm::vec4 albedo{ 1.0f, 1.0f, 1.0f, 0.0f };

			int albedo_idx = -1;
			int metallic_idx = -1;
			int normal_idx = -1;
			int roughness_idx = -1;

			int ao_idx = -1;
			int height_idx = -1;
			float metallic_factor = 0.0f;
			float roughness_factor = 1.0f;

			float ao_factor = 1.0f;
			float height_factor = 0.0f;
			float sheen_weight_factor = 0.7f;
			float sheen_roughness_factor = 1.0f;

			uint32_t albedo_enable = 0;
			uint32_t metallic_enable = 0;
			uint32_t normal_enable = 0;
			uint32_t roughness_enable = 0;

			uint32_t ao_enable = 0;
			uint32_t height_enable = 0;
			uint32_t p3;
			uint32_t p4;
		} cloth;
		static_assert(sizeof(Cloth) % 16 == 0, "std140 must be 16-byte aligned.");

	} ubo_datas_;

	std::vector<vk::raii::CommandBuffer> cmds_;

	vku::PolygonMode polygon_mode_ = vku::PolygonMode::SOLID;

	float pass_total_time_ = 0.0f;

private:
	Context& context_;
	Swapchain& swapchain_;
	TextureManager& texture_manager_;
	ModelManager& model_manager_;
	ParticleManager& particle_manager_;

	vk::raii::QueryPool timestamp_pool_{ nullptr };
	uint32_t timestamp_steps_ = 0;

	vku::Count counts_;

	uint32_t frame_counter_ = 0;
	bool first_frame_ = (frame_counter_ == 0);

	vk::SampleCountFlagBits msaa_samples_ = vk::SampleCountFlagBits::e1;

	vk::raii::DescriptorPool descriptor_pool_{ nullptr };

	struct PushConstant {
		struct ClothRender {
			glm::vec4 color;
			uint32_t nx1;
			uint32_t ny1;
			uint32_t p0;
			float p1;
		} cloth_render;
		static_assert(sizeof(ClothRender) % 4 == 0, "push constant must be multiple of 4 bytes");

		struct SoftBody {
			glm::vec4 color;
		} softbody;
		static_assert(sizeof(SoftBody) % 4 == 0, "push constant must be multiple of 4 bytes");
	} push_constants_;

	struct UBO {
		vk::raii::Buffer global{ nullptr };
		vk::raii::Buffer object{ nullptr };
		vk::raii::Buffer light{ nullptr };
		vk::raii::Buffer skybox{ nullptr };
		vk::raii::Buffer cloth{ nullptr };
	} ubos_;

	struct UBOMemory {
		vk::raii::DeviceMemory global{ nullptr };
		vk::raii::DeviceMemory object{ nullptr };
		vk::raii::DeviceMemory light{ nullptr };
		vk::raii::DeviceMemory skybox{ nullptr };
		vk::raii::DeviceMemory cloth{ nullptr };
	} ubo_memories_;

	struct UBOMapped {
		void* global{ nullptr };
		void* object{ nullptr };
		void* light{ nullptr };
		void* skybox{ nullptr };
		void* cloth{ nullptr };
	} ubo_mapped_;

	struct UBOSize {
		vk::DeviceSize global;
		vk::DeviceSize object;
		vk::DeviceSize light;
		vk::DeviceSize skybox;
		vk::DeviceSize cloth;
	} ubo_size_;

	struct SetLayout {
		vk::raii::DescriptorSetLayout global{ nullptr };
		vk::raii::DescriptorSetLayout object{ nullptr };
		vk::raii::DescriptorSetLayout lighting{ nullptr };
		vk::raii::DescriptorSetLayout skybox{ nullptr };
		vk::raii::DescriptorSetLayout cloth{ nullptr };
		vk::raii::DescriptorSetLayout softbody{ nullptr };
	} set_layouts_;

	struct Set {
		vk::raii::DescriptorSet global{ nullptr };
		vk::raii::DescriptorSet object{ nullptr };
		vk::raii::DescriptorSet lighting{ nullptr };
		vk::raii::DescriptorSet skybox{ nullptr };
		vk::raii::DescriptorSet cloth{ nullptr };
		vk::raii::DescriptorSet softbody{ nullptr };
	} sets_;

	struct PipelineLayout {
		vk::raii::PipelineLayout model{ nullptr };
		vk::raii::PipelineLayout lighting{ nullptr };
		vk::raii::PipelineLayout skybox{ nullptr };
		vk::raii::PipelineLayout cloth{ nullptr };
		vk::raii::PipelineLayout softbody{ nullptr };
	} pipeline_layouts_;

	struct Pipeline {
		vk::raii::Pipeline model_solid{ nullptr };
		vk::raii::Pipeline model_wireframe{ nullptr };
		vk::raii::Pipeline model_point{ nullptr };
		vk::raii::Pipeline lighting{ nullptr };
		vk::raii::Pipeline skybox{ nullptr };

		vk::raii::Pipeline cloth_solid{ nullptr };
		vk::raii::Pipeline cloth_wireframe{ nullptr };
		vk::raii::Pipeline cloth_point{ nullptr };
		vk::raii::Pipeline softbody{ nullptr };
	} pipelines_;

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
	void CreateCommandBuffers();
	void CreateQueryPool();
	void CreateDescriptorSetLayout();
	void CreateDescriptorPools();
	void CreateUniformBuffers();
	void CreateDescriptorSets();
	void CreateGeometryBuffers();
	void CreateGraphicsPipelines();
};