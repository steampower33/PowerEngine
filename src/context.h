#pragma once

class Swapchain;
struct Vertex;
struct Camera;
class Model;
class Texture2D;

class Context
{
public:
	Context(GLFWwindow* glfwWindow, uint32_t width, uint32_t height);
	Context(const Context& rhs) = delete;
	Context(Context&& rhs) = delete;
	Context& operator=(const Context& rhs) = delete;
	Context& operator=(Context&& rhs) = delete;
	~Context();

	void Draw(Camera& camera, float dt);
	void WaitIdle();

private:
	GLFWwindow* glfw_window_;

	vk::raii::Context                context_;
	vk::raii::Instance               instance_{ nullptr };
	vk::raii::DebugUtilsMessengerEXT debug_messenger_{ nullptr };
	vk::raii::SurfaceKHR             surface_{ nullptr };
	vk::raii::PhysicalDevice         physical_device_{ nullptr };
	vk::SampleCountFlagBits			 msaa_samples_ = vk::SampleCountFlagBits::e1;
	vk::raii::Device                 device_{ nullptr };

	uint32_t                         queue_index_ = ~0;
	vk::raii::Queue                  queue_{ nullptr };

	vk::raii::CommandPool			 command_pool_{ nullptr };

	vk::raii::DescriptorPool		 descriptor_pool_{ nullptr };
	vk::raii::DescriptorPool		 imgui_pool_{ nullptr };

	std::unique_ptr<Swapchain>       swapchain_{ nullptr };

	vk::raii::Semaphore semaphore_{ nullptr };
	uint64_t timeline_value_{ 0 };
	std::vector<vk::raii::Fence> in_flight_fences_;
	uint32_t current_frame_{ 0 };
	uint32_t read_set_{ 0 };

	bool framebuffer_resized_{ false };

	std::vector<const char*> required_device_extension_ = {
		vk::KHRSwapchainExtensionName,
		vk::KHRSpirv14ExtensionName,
		vk::KHRSynchronization2ExtensionName,
		vk::KHRCreateRenderpass2ExtensionName
	};

private:
	struct Counts {
		uint32_t ubo = 0;
		uint32_t sb = 0;
		uint32_t sampler = 0;
		uint32_t layout = 0;
	} counts_;

	// The cloth is made from a grid of particles
	struct Particle {
		glm::vec4 pos;
		glm::vec4 vel;
		glm::vec4 uv;
		glm::vec4 normal;

		static vk::VertexInputBindingDescription GetBindingDescription() {
			return { 0, sizeof(Particle), vk::VertexInputRate::eVertex };
		}

		static std::array<vk::VertexInputAttributeDescription, 2> GetAttributeDescriptions() {
			return {
				vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Particle, pos)),
				vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Particle, uv))
			};
		}
	};

	// Cloth definition parameters
	struct Cloth {
		glm::uvec2 gridsize{ 60, 60 };
		glm::vec2 size{ 5.0f, 5.0f };
	} cloth_;

	struct ParticleDatas {
		vk::raii::Buffer input{ nullptr };
		vk::raii::DeviceMemory input_memory{ nullptr };
		vk::raii::Buffer output{ nullptr };
		vk::raii::DeviceMemory output_memory{ nullptr };
		vk::raii::Buffer index_buffer = nullptr;
		vk::raii::DeviceMemory index_buffer_memory = nullptr;
		uint32_t index_count = 0;
	} particle_datas_;

	struct Graphics {
		struct UniformData {
			glm::mat4 model;
			glm::mat4 view;
			glm::mat4 proj;
		} uniform_data;
		std::vector<vk::raii::Buffer> uniform_buffers;
		std::vector<vk::raii::DeviceMemory> uniform_buffers_memory;
		std::vector<void*> uniform_buffers_mapped;

		vk::raii::DescriptorSetLayout descriptor_set_layout{ nullptr };
		std::vector<vk::raii::DescriptorSet> descriptor_sets;

		struct PipelineLayouts {
			vk::raii::PipelineLayout sphere{ nullptr };
			vk::raii::PipelineLayout cloth{ nullptr };
		} pipeline_layouts;
		struct Pipelines {
			vk::raii::Pipeline sphere{ nullptr };
			vk::raii::Pipeline cloth{ nullptr };
		} pipelines;

		std::vector<vk::raii::CommandBuffer> command_buffers;
	} graphics_;

	struct Compute {
		struct UniformData {
			float deltaT{ 0.0f };
			// These arguments define the spring setup for the cloth piece
			// Changing these changes how the cloth reacts
			float particleMass{ 0.1f };
			float springStiffness{ 1000.0f };
			float damping{ 0.25f };
			float restDistH{ 0 };
			float restDistV{ 0 };
			float restDistD{ 0 };
			float sphereRadius{ 1.0f };
			glm::vec4 spherePos{ 0.0f, 0.0f, 0.0f, 0.0f };
			glm::vec4 gravity{ 0.0f, -9.8f, 0.0f, 0.0f };
			glm::ivec2 particleCount{ 0 };
		} uniform_data;

		std::vector<vk::raii::Buffer> uniform_buffers;
		std::vector<vk::raii::DeviceMemory> uniform_buffers_memory;
		std::vector<void*> uniform_buffers_mapped;

		vk::raii::DescriptorSetLayout descriptor_set_layout{ nullptr };
		std::vector<vk::raii::DescriptorSet> descriptor_sets;

		vk::raii::PipelineLayout pipeline_layout{ nullptr };
		vk::raii::Pipeline pipeline{ nullptr };

		std::vector<vk::raii::CommandBuffer> command_buffers;
	} compute_;

	uint32_t particle_count{ 256 };
	std::unique_ptr<Model> sphere_{ nullptr };
	std::unique_ptr<Texture2D> texture_{ nullptr };

private:
	vk::raii::Image depth_image_ = nullptr;
	vk::raii::DeviceMemory depth_image_memory_ = nullptr;
	vk::raii::ImageView depth_image_view_ = nullptr;

private:
	void DrawImgui();

	void UpdateComputeUBO();
	void RecordComputeCommandBuffer();
	void AddComputeToComputeBarrier(const vk::raii::CommandBuffer& cmd, vk::Buffer buffer);
	void AddGraphicsToComputeBarrier(const vk::raii::CommandBuffer& cmd, vk::Buffer buffer);
	void AddComputeToGraphicsBarrier(const vk::raii::CommandBuffer& cmd, vk::Buffer buffer);

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
	void CreateInstance();
	std::vector<const char*> GetRequiredExtensions();
	void SetupDebugMessenger();
	static VKAPI_ATTR vk::Bool32 VKAPI_CALL DebugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*);
	void CreateSurface();
	void PickPhysicalDevice();
	vk::SampleCountFlagBits GetMaxUsableSampleCount();
	void CreateLogicalDevice();
	void CreateCommandPool();
	void CreateCommandBuffers();

	void CreateDescriptorSetLayout();
	void CreateDescriptorPools();

	void CreateUniformBuffers();
	void CreateParticleDatas();

	void CreateDescriptorSets();
	void CreateGraphicsPipelines();
	void CreateComputePipelines();
	void CreateSyncObjects();

	void CreateDepthResources();

	void SetupImgui(uint32_t width, uint32_t height);

};