#pragma once

class Swapchain;
struct Vertex;
struct Camera;
class Model;
class Texture2D;
class MouseInteractor;

#include "vulkan_utils.h"

class Context
{
public:
	Context(GLFWwindow* glfwWindow, uint32_t width, uint32_t height);
	Context(const Context& rhs) = delete;
	Context(Context&& rhs) = delete;
	Context& operator=(const Context& rhs) = delete;
	Context& operator=(Context&& rhs) = delete;
	~Context();

	void Update(Camera& camera, MouseInteractor& mouse_interactor, float dt);
	void Draw();
	void WaitIdle();

private:
	GLFWwindow* glfw_window_;

	vk::raii::Instance               instance_{ nullptr };
	vk::raii::DebugUtilsMessengerEXT debug_messenger_{ nullptr };
	vk::raii::SurfaceKHR             surface_{ nullptr };
	vk::raii::PhysicalDevice         physical_device_{ nullptr };
	vk::SampleCountFlagBits			 msaa_samples_ = vk::SampleCountFlagBits::e1;
	bool use_float_atomics_{ false };
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

// Simulation CPU
private:

	struct Edge {
		uint32_t i, j;
		float rest;
		float lambda;
	};

	std::vector<glm::vec4> positions, velocities;
	std::vector<vk::raii::Buffer>      pos_ssbo_;            // per-frame
	std::vector<vk::raii::DeviceMemory> pos_ssbo_mem_;
	std::vector<vk::raii::Buffer>      pos_staging_;         // per-frame
	std::vector<vk::raii::DeviceMemory> pos_staging_mem_;
	std::vector<void*>                  pos_staging_map_;     // per-frame
	std::vector<float> invMass;
	std::vector<Edge> edges;

	vk::raii::DescriptorSetLayout sim_cpu_descriptor_set_layout_{ nullptr };
	std::vector<vk::raii::DescriptorSet> sim_cpu_descriptor_set_;

	vk::raii::PipelineLayout sim_cpu_pipeline_layout_{ nullptr };
	vk::raii::Pipeline sim_cpu_pipeline_{ nullptr };

	std::vector<uint32_t> indices_cpu;
	vk::raii::Buffer ib{ nullptr };
	vk::raii::DeviceMemory ibm{ nullptr };

	void CreateClothData_CPU(
		int Nx, int Ny, float spacing,
		std::vector<glm::vec4>& x,
		std::vector<glm::vec4>& v,
		std::vector<float>& w,
		std::vector<Edge>& edges
	);
	void SimulateClothXPBD_CPU(
		std::vector<glm::vec4>& x,       // 현재 위치 (GPU로 보낼 position)
		std::vector<glm::vec4>& v,       // 속도
		std::vector<float>& w,           // inverse mass (0이면 고정)
		std::vector<Edge>& edges,        // 거리 제약
		float dt,
		const glm::vec3& gravity,
		int iterations,
		float compliance,
		float damping,
		const glm::vec3& sphereCenter,
		float sphereRadius
	);

private:

	vku::Counts counts_;

	// |===== Push Constant =====|
	struct ClothPC {
		uint32_t Nx;
		uint32_t Ny;
	} cloth_pc_;
	static_assert(sizeof(ClothPC) % 4 == 0, "push constant must be multiple of 4 bytes");

	uint32_t sim_count = 0;

	// |===== Particle Info =====|
	const uint32_t Nx_ = 32;
	const uint32_t Ny_ = 32;
	const float spacing_ = 0.2;

	uint32_t particles_size_ = Nx_ * Ny_;
	uint32_t indices_size_ = 0;
	uint32_t edges_size_ = 0;

	vk::raii::Buffer positions_ssbo_{ nullptr };
	vk::raii::DeviceMemory positions_ssbo_memory_{ nullptr };
	uint32_t positions_ssbo_size_ = 0;

	vk::raii::Buffer velocities_ssbo_{ nullptr };
	vk::raii::DeviceMemory velocities_ssbo_memory_{ nullptr };
	uint32_t velocities_ssbo_size_ = 0;

	vk::raii::Buffer predicted_ssbo_{ nullptr };
	vk::raii::DeviceMemory predicted_ssbo_memory_{ nullptr };
	uint32_t predicted_ssbo_size_ = 0;

	vk::raii::Buffer inverse_mass_ssbo_{ nullptr };
	vk::raii::DeviceMemory inverse_mass_ssbo_memory_{ nullptr };
	uint32_t inverse_mass_ssbo_size_ = 0;

	vk::raii::Buffer delta_ssbo_{ nullptr };
	vk::raii::DeviceMemory delta_ssbo_memory_{ nullptr };
	uint32_t delta_ssbo_size_ = 0;

	vk::raii::Buffer dcount_ssbo_{ nullptr };
	vk::raii::DeviceMemory dcount_ssbo_memory_{ nullptr };
	uint32_t dcount_ssbo_size_ = 0;

	vk::raii::Buffer edges_ssbo_{ nullptr };
	vk::raii::DeviceMemory edges_ssbo_memory_{ nullptr };
	uint32_t edges_ssbo_size_ = 0;

	vk::raii::Buffer rest_length_ssbo_{ nullptr };
	vk::raii::DeviceMemory rest_length_ssbo_memory_{ nullptr };
	uint32_t rest_length_ssbo_size_ = 0;

	vk::raii::Buffer compliance_ssbo_{ nullptr };
	vk::raii::DeviceMemory compliance_ssbo_memory_{ nullptr };
	uint32_t compliance_ssbo_size_ = 0;

	vk::raii::Buffer lambdas_ssbo_{ nullptr };
	vk::raii::DeviceMemory lambdas_ssbo_memory_{ nullptr };
	uint32_t lambdas_ssbo_size_ = 0;

	vk::raii::Buffer particle_index_buffer_{ nullptr };
	vk::raii::DeviceMemory particle_index_buffer_memory_{ nullptr };

	// |===== Compute =====|
	struct Compute {
		struct SimParams {
			float dt;
			float inv_dt;
			float substeps;   // 정수여도 float로
			float iterations; // 정수여도 float로
			glm::vec4  gravity;    // (0,-9.81,0,0)
			uint32_t  num_particles;
			uint32_t  num_edges;
			uint32_t  _pad0;
			uint32_t  _pad1;
			float damping;            // 0~1, 예: 0.02
			float collision_friction;  // 지면 충돌시 감쇠
			float _pad2; float _pad3;
		} sim_params;

		static_assert(sizeof(SimParams) % 16 == 0, "std140 must be 16-byte aligned.");

		vk::raii::Buffer sim_params_ubo{ nullptr };
		vk::raii::DeviceMemory sim_params_ubo_memory{ nullptr };
		void* sim_params_ubo_mapped{ nullptr };
		vk::DeviceSize sim_params_slot_size;

		vk::raii::DescriptorSetLayout sim_params_set_layout{ nullptr };
		vk::raii::DescriptorSet sim_params_set{ nullptr };

		vk::raii::DescriptorSetLayout cloth_compute_set_layout{ nullptr };
		vk::raii::DescriptorSet cloth_compute_set{ nullptr };

		struct PipelineLayouts {
			vk::raii::PipelineLayout common{ nullptr };
		} pipeline_layouts;

		struct Pipelines {
			vk::raii::Pipeline integrate{ nullptr };
			vk::raii::Pipeline clear_lambdas{ nullptr };
			vk::raii::Pipeline clear_deltas{ nullptr };
			vk::raii::Pipeline solve_stretch{ nullptr };
			vk::raii::Pipeline apply_deltas{ nullptr };
			vk::raii::Pipeline collide_sphere{ nullptr };
			vk::raii::Pipeline vel_update{ nullptr };
		} pipelines;

		std::vector<vk::raii::CommandBuffer> command_buffers;
	} compute_;


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
		} object_ubo_data;
		vk::raii::Buffer object_ubo{ nullptr };
		vk::raii::DeviceMemory object_ubo_memory{ nullptr };
		void* object_ubo_mapped{ nullptr };
		vk::DeviceSize object_slot_size;

		vk::raii::DescriptorSetLayout global_set_layout{ nullptr };
		vk::raii::DescriptorSet global_set{ nullptr };
		vk::raii::DescriptorSetLayout object_set_layout{ nullptr };
		vk::raii::DescriptorSet object_set{ nullptr };
		vk::raii::DescriptorSetLayout cloth_set_layout{ nullptr };
		vk::raii::DescriptorSet cloth_set{ nullptr };

		struct PipelineLayouts {
			vk::raii::PipelineLayout model{ nullptr };
			vk::raii::PipelineLayout cloth{ nullptr };
		} pipeline_layouts;

		struct Pipelines {
			vk::raii::Pipeline model{ nullptr };
			vk::raii::Pipeline cloth{ nullptr };
		} pipelines;

		std::vector<vk::raii::CommandBuffer> command_buffers;
	} graphics_;

	// |===== Model & Texture =====|
	static constexpr uint32_t kMaxObjects = 8;
	uint32_t model_count_ = 0;
	std::vector<std::unique_ptr<Model>> models;
	std::unique_ptr<Texture2D> texture_{ nullptr };

	// |===== Depth Image =====|
	vk::raii::Image depth_image_ = nullptr;
	vk::raii::DeviceMemory depth_image_memory_ = nullptr;
	vk::raii::ImageView depth_image_view_ = nullptr;

private:
	void DrawImgui();

	void UpdateMouseInteractor(Camera& camera, MouseInteractor& mouse_interactor);
	void UpdatePushContants();
	void UpdateComputeUBO();
	void UpdateGraphicsUBO(Camera& camera);

	void RecordComputeCommandBuffer();

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
	void CreateSSBOs();

	void CreateDescriptorSets();
	void CreateComputePipelines();
	void CreateGraphicsPipelines();
	void CreateSyncObjects();

	void CreateDepthResources();

	void SetupImgui(uint32_t width, uint32_t height);

};