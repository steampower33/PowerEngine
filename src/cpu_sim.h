
class Texture2D;
class Swapchain;

class CpuSim {

public:
	CpuSim(
		vk::raii::PhysicalDevice& physicalDevice,
		vk::raii::Device& device,
		vk::raii::Queue& queue,
		vk::raii::CommandPool& commandPool,
		std::unique_ptr<Swapchain>& swapchain,
		uint32_t Nx, uint32_t Ny, float spacing, std::unique_ptr<Texture2D>& texture,
		vk::raii::DescriptorSetLayout& globalSetLayout);
	CpuSim(const CpuSim& rhs) = delete;
	CpuSim(CpuSim&& rhs) = delete;
	CpuSim& operator=(const CpuSim& rhs) = delete;
	CpuSim& operator=(CpuSim&& rhs) = delete;
	~CpuSim() = default;

	float dt_ = 1.0f / 120.0f;
	glm::vec3 gravity_ = glm::vec3(0, -9.8f, 0);
	int iterations_ = 10;
	float compliance = 1e-5f;
	float damping = 0.03f;

private:
	vku::Counts counts_;
	vk::raii::DescriptorPool descriptor_pool_{ nullptr };

	struct Edge {
		uint32_t i, j;
		float rest;
		float lambda;
	};

	uint32_t Nx_, Ny_;
	float spacing_;
	uint32_t particles_size_ = 0;
	uint32_t indices_size_ = 0;

	std::vector<glm::vec4> positions, velocities;
	std::vector<float> invMass;
	std::vector<Edge> edges;

	std::vector<vk::raii::Buffer>      pos_ssbo_;            // per-frame
	std::vector<vk::raii::DeviceMemory> pos_ssbo_mem_;
	std::vector<vk::raii::Buffer>      pos_staging_;         // per-frame
	std::vector<vk::raii::DeviceMemory> pos_staging_mem_;
	std::vector<void*>                  pos_staging_map_;     // per-frame

	vk::raii::DescriptorSetLayout sim_cpu_descriptor_set_layout_{ nullptr };
	std::vector<vk::raii::DescriptorSet> sim_cpu_descriptor_set_;

	vk::raii::PipelineLayout sim_cpu_pipeline_layout_{ nullptr };
	vk::raii::Pipeline sim_cpu_pipeline_{ nullptr };

	std::vector<uint32_t> indices_cpu;
	vk::raii::Buffer ib{ nullptr };
	vk::raii::DeviceMemory ibm{ nullptr };

	// |===== Push Constant =====|
	struct ClothPC {
		uint32_t Nx;
		uint32_t Ny;
	} cloth_pc_;
	static_assert(sizeof(ClothPC) % 4 == 0, "push constant must be multiple of 4 bytes");

public:
	void CreateClothData_CPU(
		vk::raii::PhysicalDevice& physicalDevice,
		vk::raii::Device& device,
		vk::raii::Queue& queue,
		vk::raii::CommandPool& commandPool);
	void SimulateClothXPBD_CPU(
		const glm::vec3& sphereCenter,
		float sphereRadius
	);
	void UpdatePushContants();
	void CopyPositions(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd);
	void Record(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd, vk::raii::DescriptorSet& globalSet, uint32_t globalOffset);
};