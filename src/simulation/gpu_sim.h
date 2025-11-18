#pragma once

class Context;
class Swapchain;
class Texture;
class TextureManager;
class Model;
class ModelManager;

#include "vulkan_utils.h"

class GpuSim
{
public:
	GpuSim(
		Context& context,
		Swapchain& swapchain,
		TextureManager& textureManager,
		vk::raii::DescriptorSetLayout& globalSetLayout,
		std::vector<vk::Format>& formats);
	GpuSim(const GpuSim& rhs) = delete;
	GpuSim(GpuSim&& rhs) = delete;
	GpuSim& operator=(const GpuSim& rhs) = delete;
	GpuSim& operator=(GpuSim&& rhs) = delete;
	~GpuSim() = default;

	void UpdateComputeUBO(uint32_t currentFrame, std::unique_ptr<Model>& model);
	void ComputeRecord(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd, vk::raii::QueryPool& timestampPool, uint32_t& timestampSteps, vku::TestScene& testScene);
	void GraphicsRecord(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd, vk::raii::DescriptorSet& globalSet, uint32_t globalOffset, vku::PolygonMode mode);
	void UpdateTestScene(const vk::raii::CommandBuffer& cmd, vku::TestScene& testScene);

	uint32_t timestamp_count_ = 10;

	const uint32_t Nx_ = 64;
	const uint32_t Ny_ = 64;
	const float spacing_ = 0.05;

	uint32_t particles_size_ = Nx_ * Ny_;
	uint32_t indices_size_ = 0;
	uint32_t edge_size = 0;
	uint32_t bend_size = 0;

	float deviding_dt_ = 120.0f;
	int iterations_ = 10;
	float bendCompliance = 1.0f;

	vku::Counts counts_;
	vk::raii::DescriptorPool descriptor_pool_{ nullptr };

	vk::raii::Buffer index_buffer_{ nullptr };
	vk::raii::DeviceMemory index_buffer_memory_{ nullptr };

	struct Data {
		std::vector<glm::vec4> positions;
		std::vector<glm::vec4> velocities;
		std::vector<float> inverse_mass;

		struct Edge {
			uint32_t i;
			uint32_t j;
			float    rest;
			float    lambda;
		};
		static_assert(sizeof(Edge) == 16, "Edge must be 16 bytes");

		std::vector<Edge> edges;
		std::array<uint32_t, 6> pass_offset;

		struct Bend {
			uint32_t p1, p2, p3, p4;
			float restAngle;
			float lambda;
			glm::vec2 pad;
		};
		static_assert(sizeof(Bend) == 32, "Bend must be 32 bytes");

		struct SDFCollider {
			int   type;       // 0: sphere, 1: plane, 2: capsule ...
			glm::vec3  center;
			float radius;
			glm::vec3  normal;     // plane normal 등
			glm::vec3  velocity;   // 간단히 전체 rigid body 속도 넣어도 됨
			float pad;
		};
		static_assert(sizeof(SDFCollider) == 48, "SDFCollider must be 48 bytes");
		std::vector<SDFCollider> sdfColliders;
	} datas_;

	// |===== Push Constant =====|
	struct ClothPC {
		uint32_t nx1;
		uint32_t ny1;
	} cloth_pc_;
	static_assert(sizeof(ClothPC) % 4 == 0, "push constant must be multiple of 4 bytes");

	struct UBOData {
		struct SimParams {
			alignas(4)  float dt = 0.0f;
			alignas(4)  int   numParticles;
			alignas(4)  int   numEdges;
			alignas(4)  int   windTest = 0;
			alignas(4)  float windStrength = 1.0f;
			alignas(4)  float sphereRadius;
			alignas(4)  float maxSpeed;
			alignas(4)  float damping = 1.0f;
			alignas(4)  float relaxationFactor = 0.1f;
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
	} ubo_data_;
	static_assert(sizeof(UBOData::SimParams) % 16 == 0, "std140 must be 16-byte aligned.");

	struct PushConstant {
		uint32_t base;
		uint32_t count;
		float compliance;
	} pc_;

	struct UBO {
		vk::raii::Buffer sim_params{ nullptr };
	} ubos_;

	struct UBOMemory {
		vk::raii::DeviceMemory sim_params{ nullptr };
	} ubo_memories_;

	struct UBOMapped {
		void* sim_params{ nullptr };
	} ubo_mapped_;

	struct UBOSize {
		vk::DeviceSize sim_params;
	} ubo_size_;

	struct SetLayout {
		vk::raii::DescriptorSetLayout sim_params{ nullptr };
		vk::raii::DescriptorSetLayout cloth_compute{ nullptr };
		vk::raii::DescriptorSetLayout cloth_graphics{ nullptr };
	} set_layouts_;

	struct Set {
		vk::raii::DescriptorSet sim_params{ nullptr };
		vk::raii::DescriptorSet cloth_compute{ nullptr };
		vk::raii::DescriptorSet cloth_graphics{ nullptr };
	} sets_;

	struct PipelineLayout {
		vk::raii::PipelineLayout common{ nullptr };
		vk::raii::PipelineLayout cloth_graphics{ nullptr };
	} pipeline_layouts_;

	struct Pipeline {
		vk::raii::Pipeline clear_lambdas{ nullptr };
		vk::raii::Pipeline integrate{ nullptr };
		vk::raii::Pipeline solve_coloring{ nullptr };
		vk::raii::Pipeline solve_atomic{ nullptr };
		vk::raii::Pipeline solve_bend{ nullptr };
		vk::raii::Pipeline apply_deltas{ nullptr };
		vk::raii::Pipeline collide_sdf{ nullptr };
		vk::raii::Pipeline update_velocity{ nullptr };

		vk::raii::Pipeline cloth_solid{ nullptr };
		vk::raii::Pipeline cloth_wireframe{ nullptr };
		vk::raii::Pipeline cloth_point{ nullptr };

	} pipelines_;

	struct SSBO {
		vk::raii::Buffer positions{ nullptr };
		vk::raii::Buffer velocities{ nullptr };
		vk::raii::Buffer inverse_mass{ nullptr };
		vk::raii::Buffer delta_x{ nullptr };
		vk::raii::Buffer delta_y{ nullptr };
		vk::raii::Buffer delta_z{ nullptr };
		vk::raii::Buffer dcount{ nullptr };
		vk::raii::Buffer edges{ nullptr };
		vk::raii::Buffer pred_positions{ nullptr };
		vk::raii::Buffer bends{ nullptr };
	} ssbos_;

	struct SSBOMemory {
		vk::raii::DeviceMemory positions{ nullptr };
		vk::raii::DeviceMemory velocities{ nullptr };
		vk::raii::DeviceMemory inverse_mass{ nullptr };
		vk::raii::DeviceMemory delta_x{ nullptr };
		vk::raii::DeviceMemory delta_y{ nullptr };
		vk::raii::DeviceMemory delta_z{ nullptr };
		vk::raii::DeviceMemory dcount{ nullptr };
		vk::raii::DeviceMemory edges{ nullptr };
		vk::raii::DeviceMemory pred_positions{ nullptr };
		vk::raii::DeviceMemory bends{ nullptr };

	} ssbo_memories_;

	struct SSBOSize {
		uint32_t positions = 0;
		uint32_t velocities = 0;
		uint32_t inverse_mass = 0;
		uint32_t delta_x = 0;
		uint32_t delta_y = 0;
		uint32_t delta_z = 0;
		uint32_t dcount = 0;
		uint32_t edges = 0;
		uint32_t pred_positions = 0;
		uint32_t bends = 0;
	} ssbo_size_;

	struct Staging {
		vk::raii::Buffer positions{ nullptr };
		vk::raii::Buffer velocities{ nullptr };
		vk::raii::Buffer inverse_mass{ nullptr };
		vk::raii::Buffer edges{ nullptr };
	} staging_;

	struct StagingMemory {
		vk::raii::DeviceMemory positions{ nullptr };
		vk::raii::DeviceMemory velocities{ nullptr };
		vk::raii::DeviceMemory inverse_mass{ nullptr };
		vk::raii::DeviceMemory edges{ nullptr };

	} staging_memories_;

	struct StagingMapped {
		void* positions{ nullptr };
		void* velocities{ nullptr };
		void* inverse_mass{ nullptr };
		void* edges{ nullptr };
	} staging_mapped_;

};