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

	void UpdateGraphicsUBO(uint32_t currentFrame, Camera& camera, bool paused);
	void RecordGraphicsCommandBuffer(uint32_t imageIndex, uint32_t currentFrame, vku::CpuOrGpu cpuOrGpu);
	void CreateDepthResources();
	void CreateShadowResources();
	
	void CalculateGpuTime();

	std::vector<vk::raii::CommandBuffer> cmds_;

	vku::PolygonMode polygon_mode_ = vku::PolygonMode::SOLID;

	float pass_total_time_ = 0.0f;

	struct UBOData {
		ubo_data::Global global;
		ubo_data::Light light;
		ubo_data::SkyBox skybox;
		ubo_data::Shadow shadow;
	} ubo_datas_;

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
			uint32_t offset_particle;
			float p1;
		} cloth_render;
		static_assert(sizeof(ClothRender) % 4 == 0, "push constant must be multiple of 4 bytes");

		struct SoftBody {
			glm::vec4 color;
		} softbody;
		static_assert(sizeof(SoftBody) % 4 == 0, "push constant must be multiple of 4 bytes");

	} push_constants_;

	struct ShadowMap {
		glm::mat4 light_view_proj;
		uint32_t is_vertex_ssbo;
		float p0;
		float p1;
		float p2;
	} shadow_map;

	struct UBO {
		vk::raii::Buffer global{ nullptr };
		vk::raii::Buffer model{ nullptr };
		vk::raii::Buffer light{ nullptr };
		vk::raii::Buffer skybox{ nullptr };
		vk::raii::Buffer cloth{ nullptr };
		vk::raii::Buffer softbody{ nullptr };
		vk::raii::Buffer skinned_model{ nullptr };
	} ubos_;

	struct UBOMemory {
		vk::raii::DeviceMemory global{ nullptr };
		vk::raii::DeviceMemory model{ nullptr };
		vk::raii::DeviceMemory light{ nullptr };
		vk::raii::DeviceMemory skybox{ nullptr };
		vk::raii::DeviceMemory cloth{ nullptr };
		vk::raii::DeviceMemory softbody{ nullptr };
		vk::raii::DeviceMemory skinned_model{ nullptr };
	} ubo_memories_;

	struct UBOMapped {
		void* global{ nullptr };
		void* model{ nullptr };
		void* light{ nullptr };
		void* skybox{ nullptr };
		void* cloth{ nullptr };
		void* softbody{ nullptr };
		void* skinned_model{ nullptr };
	} ubo_mapped_;

	struct UBOSize {
		vk::DeviceSize global;
		vk::DeviceSize model;
		vk::DeviceSize light;
		vk::DeviceSize skybox;
		vk::DeviceSize cloth;
		vk::DeviceSize softbody;
		vk::DeviceSize skinned_model;
	} ubo_size_;

	struct SetLayout {
		vk::raii::DescriptorSetLayout global{ nullptr };
		vk::raii::DescriptorSetLayout model{ nullptr };
		vk::raii::DescriptorSetLayout lighting{ nullptr };
		vk::raii::DescriptorSetLayout skybox{ nullptr };
		vk::raii::DescriptorSetLayout cloth{ nullptr };
		vk::raii::DescriptorSetLayout softbody{ nullptr };
		vk::raii::DescriptorSetLayout skinned_model{ nullptr };
		vk::raii::DescriptorSetLayout shadow_particle{ nullptr };
	} set_layouts_;

	struct Set {
		vk::raii::DescriptorSet global{ nullptr };
		vk::raii::DescriptorSet model{ nullptr };
		vk::raii::DescriptorSet lighting{ nullptr };
		vk::raii::DescriptorSet skybox{ nullptr };
		vk::raii::DescriptorSet cloth{ nullptr };
		vk::raii::DescriptorSet softbody{ nullptr };
		vk::raii::DescriptorSet skinned_model{ nullptr };
		vk::raii::DescriptorSet shadow_particle{ nullptr };
	} sets_;

	struct PipelineLayout {
		vk::raii::PipelineLayout model{ nullptr };
		vk::raii::PipelineLayout lighting{ nullptr };
		vk::raii::PipelineLayout skybox{ nullptr };
		vk::raii::PipelineLayout cloth{ nullptr };
		vk::raii::PipelineLayout softbody{ nullptr };
		vk::raii::PipelineLayout skinned_model{ nullptr };
		vk::raii::PipelineLayout debug_capsule{ nullptr };
		vk::raii::PipelineLayout shadow_model{ nullptr };
		vk::raii::PipelineLayout shadow_particle{ nullptr };
		vk::raii::PipelineLayout infinite_grid{ nullptr };
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

		vk::raii::Pipeline softbody_solid{ nullptr };
		vk::raii::Pipeline softbody_wireframe{ nullptr };
		vk::raii::Pipeline softbody_point{ nullptr };

		vk::raii::Pipeline skinned_model_solid{ nullptr };
		vk::raii::Pipeline skinned_model_wireframe{ nullptr };
		vk::raii::Pipeline skinned_model_point{ nullptr };

		vk::raii::Pipeline debug_capsule{ nullptr };

		vk::raii::Pipeline shadow_model{ nullptr };
		vk::raii::Pipeline shadow_particle{ nullptr };

		vk::raii::Pipeline infinite_grid{ nullptr };
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

		vk::raii::Image coat_fuzz_image = nullptr;
		vk::raii::DeviceMemory coat_fuzz_image_memory = nullptr;
		vk::raii::ImageView coat_fuzz_image_view = nullptr;

	} geometry_buffers_;

	vk::raii::Image depth_image_ = nullptr;
	vk::raii::DeviceMemory depth_image_memory_ = nullptr;
	vk::raii::ImageView depth_image_view_ = nullptr;

	vk::raii::Image shadow_image_ = nullptr;
	vk::raii::DeviceMemory shadow_image_memory_ = nullptr;
	vk::raii::ImageView shadow_image_view_ = nullptr;
	vk::raii::Sampler shadow_sampler = nullptr;
	vk::Extent2D shadow_extent_{ 2048, 2048 };

private:
	void CreateCommandBuffers();
	void CreateQueryPool();
	void CreateDescriptorSetLayout();
	void CreateDescriptorPools();
	void CreateUniformBuffers();
	void CreateDescriptorSets();
	void CreateGeometryBuffers();
	void CreateGraphicsPipelines();

	void ShadowDepthOnlyPass(const vk::raii::CommandBuffer& cmd, uint32_t currentFrame);
	void PreMainRenderPass(const vk::raii::CommandBuffer& cmd, uint32_t currentFrame);
	void MainRenderPass(const vk::raii::CommandBuffer& cmd, uint32_t currentFrame);
	void PostMainRenderPass(const vk::raii::CommandBuffer& cmd, uint32_t imageIndex);
	void LightingPass(const vk::raii::CommandBuffer& cmd, uint32_t imageIndex, uint32_t currentFrame);
};