
class Texture2D;
class Swapchain;

class GpuSim
{
public:
	GpuSim(
		vk::raii::PhysicalDevice& physicalDevice,
		vk::raii::Device& device,
		vk::raii::Queue& queue,
		vk::raii::CommandPool& commandPool,
		std::unique_ptr<Swapchain>& swapchain,
		uint32_t Nx, uint32_t Ny, float spacing, std::unique_ptr<Texture2D>& texture,
		vk::raii::DescriptorSetLayout& globalSetLayout);
	GpuSim(const GpuSim& rhs) = delete;
	GpuSim(GpuSim&& rhs) = delete;
	GpuSim& operator=(const GpuSim& rhs) = delete;
	GpuSim& operator=(GpuSim&& rhs) = delete;
	~GpuSim() = default;

	void UpdateComputeUBO(uint32_t currentFrame);
	void Record(uint32_t currentFrame);

private:
	vku::Counts counts_;
	vk::raii::DescriptorPool descriptor_pool_{ nullptr };

	uint32_t Nx_ = 0;
	uint32_t Ny_ = 0;
	float spacing_ = 0;

	uint32_t particles_size_ = 0;
	uint32_t indices_size_ = 0;
	uint32_t edges_size_ = 0;

	// |===== Push Constant =====|
	struct ClothPC {
		uint32_t Nx;
		uint32_t Ny;
	} cloth_pc_;
	static_assert(sizeof(ClothPC) % 4 == 0, "push constant must be multiple of 4 bytes");

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
		vk::raii::DescriptorSetLayout cloth_set_layout{ nullptr };
		vk::raii::DescriptorSet cloth_set{ nullptr };

		struct PipelineLayouts {
			vk::raii::PipelineLayout cloth{ nullptr };
		} pipeline_layouts;

		struct Pipelines {
			vk::raii::Pipeline cloth{ nullptr };
		} pipelines;

		std::vector<vk::raii::CommandBuffer> command_buffers;
	} graphics_;

	vk::raii::Buffer positions_ssbo_{ nullptr };
	vk::raii::DeviceMemory positions_ssbo_memory_{ nullptr };
	uint32_t positions_ssbo_size_ = 0;

	vk::raii::Buffer velocities_ssbo_{ nullptr };
	vk::raii::DeviceMemory velocities_ssbo_memory_{ nullptr };
	uint32_t velocities_ssbo_size_ = 0;

	vk::raii::Buffer inverse_mass_ssbo_{ nullptr };
	vk::raii::DeviceMemory inverse_mass_ssbo_memory_{ nullptr };
	uint32_t inverse_mass_ssbo_size_ = 0;

	vk::raii::Buffer delta_x_ssbo_{ nullptr };
	vk::raii::DeviceMemory delta_x_ssbo_memory_{ nullptr };
	uint32_t delta_x_ssbo_size_ = 0;

	vk::raii::Buffer delta_y_ssbo_{ nullptr };
	vk::raii::DeviceMemory delta_y_ssbo_memory_{ nullptr };
	uint32_t delta_y_ssbo_size_ = 0;

	vk::raii::Buffer delta_z_ssbo_{ nullptr };
	vk::raii::DeviceMemory delta_z_ssbo_memory_{ nullptr };
	uint32_t delta_z_ssbo_size_ = 0;

	vk::raii::Buffer dcount_ssbo_{ nullptr };
	vk::raii::DeviceMemory dcount_ssbo_memory_{ nullptr };
	uint32_t dcount_ssbo_size_ = 0;

	struct Edge { uint32_t i; uint32_t j; float rest; float lambda; };
	vk::raii::Buffer edges_ssbo_{ nullptr };
	vk::raii::DeviceMemory edges_ssbo_memory_{ nullptr };
	uint32_t edges_ssbo_size_ = 0;

	vk::raii::Buffer index_buffer_{ nullptr };
	vk::raii::DeviceMemory index_buffer_memory_{ nullptr };
};