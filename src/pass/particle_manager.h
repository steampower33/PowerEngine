#pragma once

class Context;

// Cloth
struct Cloth
{
	float spacing = 0.015f;
	float gsm = 0.2f;
	glm::vec2 cloth_size;
	uint32_t nx = 0.0f;
	uint32_t ny = 0.0f;
	uint32_t nx1 = 0.0f;
	uint32_t ny1 = 0.0f;
	float height;

	glm::vec4 color;

	glm::vec3 origin = glm::vec3(0.0f);
	float     angle_deg = 0.0f;
	glm::vec3 axis = glm::vec3(1, 0, 0);

	uint32_t offset_particle = 0;
	uint32_t offset_indices = 0;
	uint32_t offset_stretch = 0;
	uint32_t offset_shear = 0;
	uint32_t offset_bend = 0;
	uint32_t offset_area = 0;

	uint32_t num_particle = 0;
	uint32_t num_indices = 0;
	uint32_t num_stretch = 0;
	uint32_t num_shear = 0;
	uint32_t num_bend = 0;
	uint32_t num_area = 0;
};

class ParticleManager
{
public:
	ParticleManager(Context& context);
	ParticleManager(const ParticleManager& rhs) = delete;
	ParticleManager(ParticleManager&& rhs) = delete;
	ParticleManager& operator=(const ParticleManager& rhs) = delete;
	ParticleManager& operator=(ParticleManager&& rhs) = delete;
	~ParticleManager();

	void SetCloth(Cloth& cloth);
	void Reset(Cloth& cloth);

	Context& context_;

	uint32_t total_particles_ = 0;
	uint32_t total_indices_ = 0;

	std::vector<Cloth> clothes_;

	std::vector<glm::vec4> positions;
	std::vector<glm::vec4> pred_positions;
	std::vector<glm::vec4> velocities;
	std::vector<float> inverse_masses;
	std::vector<float> masses;
	std::vector<uint32_t> indices;

	vk::raii::Buffer index_buffer_{ nullptr };
	vk::raii::DeviceMemory index_buffer_memory_{ nullptr };

	std::vector<uint32_t> particle_hashes;
	std::vector<uint32_t> particle_indices;
	std::vector<uint32_t> starts;
	std::vector<uint32_t> ends;

	uint32_t num_neighbors;
	std::vector<uint32_t> neighbors;
	std::vector<uint32_t> neighbor_lambdas;

	struct SSBO {
		vk::raii::Buffer position{ nullptr };
		vk::raii::Buffer pred_position{ nullptr };
		vk::raii::Buffer velocity{ nullptr };
		vk::raii::Buffer inverse_mass{ nullptr };

		vk::raii::Buffer particle_hash{ nullptr };
		vk::raii::Buffer particle_indice{ nullptr };
		vk::raii::Buffer start{ nullptr };
		vk::raii::Buffer end{ nullptr };
		vk::raii::Buffer neighbor{ nullptr };
		vk::raii::Buffer neighbor_lambda{ nullptr };
	} ssbos_;

	struct SSBOMemory {
		vk::raii::DeviceMemory position{ nullptr };
		vk::raii::DeviceMemory pred_position{ nullptr };
		vk::raii::DeviceMemory velocity{ nullptr };
		vk::raii::DeviceMemory inverse_mass{ nullptr };

		vk::raii::DeviceMemory particle_hash{ nullptr };
		vk::raii::DeviceMemory particle_indice{ nullptr };
		vk::raii::DeviceMemory start{ nullptr };
		vk::raii::DeviceMemory end{ nullptr };
		vk::raii::DeviceMemory neighbor{ nullptr };
		vk::raii::DeviceMemory neighbor_lambda{ nullptr };
	} ssbo_memories_;

	struct SSBOSize {
		vk::DeviceSize position = 0;
		vk::DeviceSize pred_position = 0;
		vk::DeviceSize velocity = 0;
		vk::DeviceSize inverse_mass = 0;

		vk::DeviceSize particle_hash = 0;
		vk::DeviceSize particle_indice = 0;
		vk::DeviceSize start = 0;
		vk::DeviceSize end = 0;
		vk::DeviceSize neighbor = 0;
		vk::DeviceSize neighbor_lambda = 0;
	} ssbo_size_;

	struct Staging {
		vk::raii::Buffer position{ nullptr };
		vk::raii::Buffer pred_position{ nullptr };
		vk::raii::Buffer velocity{ nullptr };
		vk::raii::Buffer inverse_mass{ nullptr };
	} staging_;

	struct StagingMemory {
		vk::raii::DeviceMemory position{ nullptr };
		vk::raii::DeviceMemory pred_position{ nullptr };
		vk::raii::DeviceMemory velocity{ nullptr };
		vk::raii::DeviceMemory inverse_mass{ nullptr };
	} staging_memories_;

	struct StagingMapped {
		void* position{ nullptr };
		void* pred_position{ nullptr };
		void* velocity{ nullptr };
		void* inverse_mass{ nullptr };
	} staging_mapped_;

private:
	void CreateSSBO();
};