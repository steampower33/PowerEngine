#pragma once

class Context;
class Swapchain;
class Texture2D;
class TextureManager;
class Model;
class ModelManager;

class GpuSim
{
public:
	GpuSim(
		Context& context,
		Swapchain& swapchain,
		TextureManager& textureManager,
		vk::raii::DescriptorSetLayout& globalSetLayout,
		uint32_t Nx, uint32_t Ny, float spacing);
	GpuSim(const GpuSim& rhs) = delete;
	GpuSim(GpuSim&& rhs) = delete;
	GpuSim& operator=(const GpuSim& rhs) = delete;
	GpuSim& operator=(GpuSim&& rhs) = delete;
	~GpuSim() = default;

	void UpdateComputeUBO(uint32_t currentFrame, std::unique_ptr<Model>& model);
	void ComputeRecord(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd, vk::raii::QueryPool& timestampPool, uint32_t& timestampSteps, vku::TestScene& testScene);
	void GraphicsRecord(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd, vk::raii::DescriptorSet& globalSet, uint32_t globalOffset);
	void UpdateTestScene(const vk::raii::CommandBuffer& cmd, vku::TestScene& testScene);

	uint32_t iter_contraint_count_ = 10;

	uint32_t Nx_ = 0;
	uint32_t Ny_ = 0;
	float spacing_ = 0;

	uint32_t particles_size_ = 0;
	uint32_t indices_size_ = 0;

	float dt_ = 1 / 60.0f;
	int iterations_ = 10;

	// |===== Push Constant =====|
	struct ClothPC {
		uint32_t Nx;
		uint32_t Ny;
	} cloth_pc_;
	static_assert(sizeof(ClothPC) % 4 == 0, "push constant must be multiple of 4 bytes");

	vku::Counts counts_;
	vk::raii::DescriptorPool descriptor_pool_{ nullptr };
	
	// |===== Compute =====|
	struct Compute {
		struct SimParams {
			alignas(4)  float dt = 0.0f;
			alignas(4)  int   numParticles;
			alignas(4)  int   numEdges;
			alignas(4)  int   windTest = 0;
			alignas(4)  float windStrength = 1.0f;
			alignas(4)  float sphereRadius;
			alignas(4)  float maxSpeed;
			alignas(4)  float damping = 0.2f;
			alignas(4)  float relaxationFactor = 0.2f;
			alignas(4)  int   numBends;
			alignas(4)  uint32_t numColliders = 1;
			alignas(4)  float collisionMargin = 0.1f;
			alignas(16) glm::vec4 sphereCenter;
			alignas(16) glm::vec4 windDir = glm::vec4(0.0f, 0.0f, -1.0f, 0.0f);
			alignas(16) glm::vec4 gravity = glm::vec4(0.0f, -9.8f, 0.0f, 0.0f);
			alignas(4) float thickness = 0.004f;
			alignas(4) float friction = 0.01f;
			alignas(4) float pad1;
			alignas(4) float pad2;
		} sim_params;

		static_assert(sizeof(SimParams) % 16 == 0, "std140 must be 16-byte aligned.");

		struct PC {
			uint32_t base;
			uint32_t count;
			float compliance;
		} pc_;

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
			vk::raii::Pipeline integrate{ nullptr };
			vk::raii::Pipeline solve_coloring{ nullptr };
			vk::raii::Pipeline solve_atomic{ nullptr };
			vk::raii::Pipeline solve_bend{ nullptr };
			vk::raii::Pipeline apply_deltas{ nullptr };
			vk::raii::Pipeline collide_sdf{ nullptr };
			vk::raii::Pipeline update_velocity{ nullptr };
		} pipelines;

	} compute_;

	// |===== Graphics Info =====|
	struct Graphics {
		vk::raii::DescriptorSetLayout cloth_set_layout{ nullptr };
		vk::raii::DescriptorSet cloth_set{ nullptr };

		struct PipelineLayouts {
			vk::raii::PipelineLayout cloth_solid{ nullptr };
		} pipeline_layouts;

		struct Pipelines {
			vk::raii::Pipeline cloth_solid{ nullptr };
			vk::raii::Pipeline cloth_wireframe{ nullptr };
			vk::raii::Pipeline cloth_point{ nullptr };
		} pipelines;

	} graphics_;

	bool is_wireframe_ = false;
	bool is_point_ = false;

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

	vk::raii::Buffer edges_ssbo_{ nullptr };
	vk::raii::DeviceMemory edges_ssbo_memory_{ nullptr };
	uint32_t edges_ssbo_size_ = 0;
	vk::raii::Buffer edges_staging_{ nullptr };
	vk::raii::DeviceMemory edges_staging_memory_{ nullptr };
	void* edges_staging_mapped_{ nullptr };

	vk::raii::Buffer pred_positions_ssbo_{ nullptr };
	vk::raii::DeviceMemory pred_positions_ssbo_memory_{ nullptr };
	uint32_t pred_positions_ssbo_size_ = 0;

	vk::raii::Buffer bends_ssbo_{ nullptr };
	vk::raii::DeviceMemory bends_ssbo_memory_{ nullptr };
	uint32_t bends_ssbo_size_ = 0;

	vk::raii::Buffer index_buffer_{ nullptr };
	vk::raii::DeviceMemory index_buffer_memory_{ nullptr };

	// Edge
	struct Edge {
		uint32_t i;
		uint32_t j;
		float    rest;
		float    lambda;
	};
	static_assert(sizeof(Edge) == 16, "Edge must be 16 bytes");

	uint32_t edge_size_;
	std::vector<Edge> edges_;
	std::array<uint32_t, 6> pass_offset_;

	struct Bend {
		uint32_t p1, p2, p3, p4;
		float restAngle;
		float lambda;
		glm::vec2 pad;
	};
	static_assert(sizeof(Bend) == 32, "Bend must be 32 bytes");
	uint32_t bends_size_ = 0;

	struct SDFCollider {
		int   type;       // 0: sphere, 1: plane, 2: capsule ...
		glm::vec3  center;
		float radius;
		glm::vec3  normal;     // plane normal 등
		glm::vec3  velocity;   // 간단히 전체 rigid body 속도 넣어도 됨
		float pad;
	};
	std::vector<SDFCollider> sdfColliders;
};