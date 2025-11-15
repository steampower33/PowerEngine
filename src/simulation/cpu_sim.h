#pragma once

class Context;
class Swapchain;
class ModelManager;
class Texture2D;
class TextureManager;

class CpuSim {

public:
	CpuSim(
		Context& context,
		Swapchain& swapchain,
		TextureManager& textureManager,
		vk::raii::DescriptorSetLayout& globalSetLayout,
		uint32_t Nx, uint32_t Ny, float spacing);
	CpuSim(const CpuSim& rhs) = delete;
	CpuSim(CpuSim&& rhs) = delete;
	CpuSim& operator=(const CpuSim& rhs) = delete;
	CpuSim& operator=(CpuSim&& rhs) = delete;
	~CpuSim() = default;

	float dt_ = 1.0f / 60.0f;
	glm::vec3 gravity_ = glm::vec3(0, -9.8f, 0);
	int iterations_ = 8;
	float compliance_ = 1e-6f;
	float damping_ = 0.01f;

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

	std::vector<glm::vec4> positions_, velocities_;
	std::vector<float> inv_mass_;
	std::vector<Edge> edges_;

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

private:

	uint32_t stretch_edge_size_ = 0;
	uint32_t bend_edge_size_ = 0;

	struct EdgeKey { uint32_t a, b; };               // a < b
	static inline EdgeKey makeKey(uint32_t i, uint32_t j) {
		return (i < j) ? EdgeKey{ i,j } : EdgeKey{ j,i };
	}
	struct KeyHash {
		size_t operator()(EdgeKey k) const {
			return (size_t(k.a) << 32) ^ size_t(k.b);
		}
	};
	struct KeyEq {
		bool operator()(EdgeKey x, EdgeKey y) const {
			return x.a == y.a && x.b == y.b;
		}
	};
};