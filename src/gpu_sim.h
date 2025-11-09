
class Texture2D;
class Swapchain;
class Model;

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

	void UpdateComputeUBO(uint32_t currentFrame, std::unique_ptr<Model>& model);
	void ComputeRecord(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd, vk::raii::QueryPool& timestampPool, uint32_t& steps, vku::TestScene& testScene);
	void UpdatePushContants();
	void GraphicsRecord(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd, vk::raii::DescriptorSet& globalSet, uint32_t globalOffset);

	vku::Counts counts_;
	vk::raii::DescriptorPool descriptor_pool_{ nullptr };

	uint32_t Nx_ = 0;
	uint32_t Ny_ = 0;
	float spacing_ = 0;

	uint32_t particles_size_ = 0;
	uint32_t indices_size_ = 0;
	uint32_t edges_size_ = 0;

	uint32_t iterations_ = 8;

	// |===== Push Constant =====|
	struct ClothPC {
		uint32_t Nx;
		uint32_t Ny;
	} cloth_pc_;
	static_assert(sizeof(ClothPC) % 4 == 0, "push constant must be multiple of 4 bytes");

	// |===== Compute =====|
	struct Compute {
		struct SimParams {
			alignas(4)  float dt = 1 / 480.0f;
			alignas(4)  float compliance = 1e-6f;
			alignas(4)  float damping = 0.01f;
			alignas(4)  int   numVerts;
			alignas(4)  int   numEdges;
			alignas(16) glm::vec4 gravity = glm::vec4(0.0f, -9.8f, 0.0f, 0.0f);
			alignas(16) glm::vec4 sphereCenter;
			alignas(4)  float sphereRadius;
			alignas(4)  float collisionBeta = 0.3f;
			alignas(4)  float pad0;
			alignas(4)  float pad1;
			alignas(4)  float pad2;
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
			vk::raii::Pipeline clear_lambdas{ nullptr };
			vk::raii::Pipeline copy_xprev{ nullptr };
			vk::raii::Pipeline integrate{ nullptr };
			vk::raii::Pipeline clear_deltas{ nullptr };
			vk::raii::Pipeline solve_stretch{ nullptr };
			vk::raii::Pipeline apply_deltas{ nullptr };
			vk::raii::Pipeline collide_sphere{ nullptr };
			vk::raii::Pipeline update_velocity{ nullptr };
		} pipelines;

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

	} graphics_;

	std::vector<glm::vec4> positions_;
	vk::raii::Buffer positions_ssbo_{ nullptr };
	vk::raii::DeviceMemory positions_ssbo_memory_{ nullptr };
	uint32_t positions_ssbo_size_ = 0;
	vk::raii::Buffer positions_staging_{ nullptr };
	vk::raii::DeviceMemory positions_staging_memory_{ nullptr };
	void* positions_staging_mapped_{ nullptr };

	std::vector<glm::vec4> velocities_;
	vk::raii::Buffer velocities_ssbo_{ nullptr };
	vk::raii::DeviceMemory velocities_ssbo_memory_{ nullptr };
	uint32_t velocities_ssbo_size_ = 0;
	vk::raii::Buffer velocities_staging_{ nullptr };
	vk::raii::DeviceMemory velocities_staging_memory_{ nullptr };
	void* velocities_staging_mapped_{ nullptr };

	std::vector<float> inverse_mass_;
	vk::raii::Buffer inverse_mass_ssbo_{ nullptr };
	vk::raii::DeviceMemory inverse_mass_ssbo_memory_{ nullptr };
	uint32_t inverse_mass_ssbo_size_ = 0;
	vk::raii::Buffer inverse_mass_staging_{ nullptr };
	vk::raii::DeviceMemory inverse_mass_staging_memory_{ nullptr };
	void* inverse_mass_staging_mapped_{ nullptr };

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

	vk::raii::Buffer prev_positions_ssbo_{ nullptr };
	vk::raii::DeviceMemory prev_positions_ssbo_memory_{ nullptr };
	uint32_t prev_positions_ssbo_size_ = 0;

	vk::raii::Buffer index_buffer_{ nullptr };
	vk::raii::DeviceMemory index_buffer_memory_{ nullptr };
};