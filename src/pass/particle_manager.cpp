
#include "context.h"
#include "vulkan_utils.h"

#include "particle_manager.h"

ParticleManager::ParticleManager(Context& context)
	: context_(context)
{
	{
		Cloth cloth;
		//cloth.spacing = 0.1f;
		cloth.cloth_size = glm::vec2(1.0f, 1.0f);
		cloth.nx = (uint32_t)std::round(cloth.cloth_size.x / cloth.spacing);
		cloth.ny = (uint32_t)std::round(cloth.cloth_size.y / cloth.spacing);
		cloth.nx1 = cloth.nx + 1;
		cloth.ny1 = cloth.ny + 1;
		cloth.num_particle = cloth.nx1 * cloth.ny1;
		cloth.height = 2.0f;

		SetCloth(cloth);

		clothes_.push_back(cloth);
	}

	{
		Cloth cloth;
		cloth.cloth_size = glm::vec2(1.0f, 1.0f);
		cloth.nx = (uint32_t)std::round(cloth.cloth_size.x / cloth.spacing);
		cloth.ny = (uint32_t)std::round(cloth.cloth_size.y / cloth.spacing);
		cloth.nx1 = cloth.nx + 1;
		cloth.ny1 = cloth.ny + 1;
		cloth.num_particle = cloth.nx1 * cloth.ny1;
		cloth.height = 4.0f;

		SetCloth(cloth);

		clothes_.push_back(cloth);
	}

	{
		Cloth cloth;
		cloth.cloth_size = glm::vec2(1.0f, 1.0f);
		cloth.nx = (uint32_t)std::round(cloth.cloth_size.x / cloth.spacing);
		cloth.ny = (uint32_t)std::round(cloth.cloth_size.y / cloth.spacing);
		cloth.nx1 = cloth.nx + 1;
		cloth.ny1 = cloth.ny + 1;
		cloth.num_particle = cloth.nx1 * cloth.ny1;
		cloth.height = 6.0f;

		SetCloth(cloth);

		clothes_.push_back(cloth);
	}

	vku::CreateIndexBuffer(context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_, indices, index_buffer_, index_buffer_memory_);

	CreateSSBO();
}

ParticleManager::~ParticleManager()
{

}

void ParticleManager::SetCloth(Cloth& cloth)
{
	const int nx = cloth.nx;
	const int ny = cloth.ny;
	const int nx1 = cloth.nx1;
	const int ny1 = cloth.ny1;

	cloth.offset_particle = total_particles_;
	total_particles_ += cloth.num_particle;

	auto vid = [&](int x, int y) { return cloth.offset_particle + uint32_t(y * nx1 + x); };

	const uint32_t N = nx1 * ny1;

	float angle_rad = glm::radians(cloth.angle_deg);
	glm::mat4 R = glm::rotate(glm::mat4(1.0f), angle_rad, glm::normalize(cloth.axis));

	// Set positions, velocities, pred_positions
	for (int y = 0; y < ny1; ++y) {
		for (int x = 0; x < nx1; ++x) {
			float lx = (-0.5f * cloth.cloth_size.x) + x * cloth.spacing;
			float ly = 0.0f; // 평면 자체는 y=0에서 시작
			float lz = (-0.5f * cloth.cloth_size.y) + y * cloth.spacing;

			glm::vec4 local(lx, ly, lz, 1.0f);

			glm::vec3 rotated = glm::vec3(R * local);

			glm::vec3 worldPos = cloth.origin + rotated + glm::vec3(0.0f, cloth.height, 0.0f);
			
            positions.push_back(glm::vec4(worldPos, 0.0f));
			velocities.push_back(glm::vec4(0));
			pred_positions.push_back(glm::vec4(worldPos, 0.0f));
			masses.push_back(0.0f);
			inverse_masses.push_back(0.0f);
		}
	}

	// Set indices
	for (int y = 0; y < ny; ++y) {
		for (int x = 0; x < nx; ++x) {
			uint32_t i0 = vid(x, y);
			uint32_t i1 = vid(x + 1, y);
			uint32_t i2 = vid(x, y + 1);
			uint32_t i3 = vid(x + 1, y + 1);
			indices.push_back(i0); indices.push_back(i2); indices.push_back(i1);
			indices.push_back(i1); indices.push_back(i2); indices.push_back(i3);
		}
	}
	cloth.offset_indices = total_indices_;
	cloth.num_indices = static_cast<uint32_t>(indices.size() - cloth.offset_indices);
	total_indices_ += cloth.num_indices;

	// Total area
	float totalArea = 0.0f;
	for (size_t t = cloth.offset_indices; t < cloth.offset_indices + cloth.num_indices; t += 3) {
		uint32_t i0 = indices[t + 0];
		uint32_t i1 = indices[t + 1];
		uint32_t i2 = indices[t + 2];

		glm::vec3 p0 = glm::vec3(positions[i0]);
		glm::vec3 p1 = glm::vec3(positions[i1]);
		glm::vec3 p2 = glm::vec3(positions[i2]);

		glm::vec3 e1 = p1 - p0;
		glm::vec3 e2 = p2 - p0;

		float area = 0.5f * glm::length(glm::cross(e1, e2)); // Triangle area
		totalArea += area;
	}

	float totalMassTarget = cloth.cloth_size.x * cloth.cloth_size.y * cloth.gsm;

	// Area zero defence
	float density = 0.0f;
	if (totalArea > 0.0f) {
		density = totalMassTarget / totalArea; // kg/m²
	}

	// Distribute mass to each triangle in proportion to area
	for (size_t t = cloth.offset_indices; t < cloth.offset_indices + cloth.num_indices; t += 3) {
		uint32_t i0 = indices[t + 0];
		uint32_t i1 = indices[t + 1];
		uint32_t i2 = indices[t + 2];

		glm::vec3 p0 = glm::vec3(positions[i0]);
		glm::vec3 p1 = glm::vec3(positions[i1]);
		glm::vec3 p2 = glm::vec3(positions[i2]);

		glm::vec3 e1 = p1 - p0;
		glm::vec3 e2 = p2 - p0;

		float area = 0.5f * glm::length(glm::cross(e1, e2));

		float triMass = density * area;

		float share = triMass / 3.0f;
		masses[i0] += share;
		masses[i1] += share;
		masses[i2] += share;
	}

	// Set inverse masses using mass
	for (uint32_t i = cloth.offset_particle; i < cloth.offset_particle + cloth.num_particle; i++) {
		float m = masses[i];
		if (m > 0.0f)
			inverse_masses[i] = 1.0f / m;
		else
			inverse_masses[i] = 0.0f;
	}

	//inverse_masses[cloth.offset_particle] = 0.0f;
	//inverse_masses[cloth.offset_particle + nx1 - 1] = 0.0f;
}

void ParticleManager::Reset(Cloth& cloth)
{

	const int nxCells = cloth.nx;
	const int nyCells = cloth.ny;
	const int nx1 = nxCells + 1;
	const int ny1 = nyCells + 1;

	auto vid = [&](int x, int y) { return cloth.offset_particle + uint32_t(y * nx1 + x); };

	const uint32_t N = nx1 * ny1;

	float angle_rad = glm::radians(cloth.angle_deg);
	glm::mat4 R = glm::rotate(glm::mat4(1.0f), angle_rad, glm::normalize(cloth.axis));

	// Set positions, velocities, pred_positions
	for (int y = 0; y < ny1; ++y) {
		for (int x = 0; x < nx1; ++x) {
			uint32_t id = vid(x, y);

			float lx = (-0.5f * cloth.cloth_size.x) + x * cloth.spacing;
			float ly = 0.0f; // 평면 자체는 y=0에서 시작
			float lz = (-0.5f * cloth.cloth_size.y) + y * cloth.spacing;

			glm::vec4 local(lx, ly, lz, 1.0f);

			glm::vec3 rotated = glm::vec3(R * local);

			glm::vec3 worldPos = cloth.origin + rotated + glm::vec3(0.0f, cloth.height, 0.0f);

			positions[id] = { worldPos, 0.0f };
			velocities[id] = glm::vec4(0.0f);
			pred_positions[id] = { worldPos, 0.0f };
		}
	}

	// Total area
	float totalArea = 0.0f;
	for (size_t t = cloth.offset_indices; t < cloth.offset_indices + cloth.num_indices; t += 3) {
		uint32_t i0 = indices[t + 0];
		uint32_t i1 = indices[t + 1];
		uint32_t i2 = indices[t + 2];

		glm::vec3 p0 = glm::vec3(positions[i0]);
		glm::vec3 p1 = glm::vec3(positions[i1]);
		glm::vec3 p2 = glm::vec3(positions[i2]);

		glm::vec3 e1 = p1 - p0;
		glm::vec3 e2 = p2 - p0;

		float area = 0.5f * glm::length(glm::cross(e1, e2)); // Triangle area
		totalArea += area;
	}

	float totalMassTarget = cloth.cloth_size.x * cloth.cloth_size.y * cloth.gsm;

	// Area zero defence
	float density = 0.0f;
	if (totalArea > 0.0f) {
		density = totalMassTarget / totalArea; // kg/m²
	}

	// Distribute mass to each triangle in proportion to area
	for (size_t t = cloth.offset_indices; t < cloth.offset_indices + cloth.num_indices; t += 3) {
		uint32_t i0 = indices[t + 0];
		uint32_t i1 = indices[t + 1];
		uint32_t i2 = indices[t + 2];

		glm::vec3 p0 = glm::vec3(positions[i0]);
		glm::vec3 p1 = glm::vec3(positions[i1]);
		glm::vec3 p2 = glm::vec3(positions[i2]);

		glm::vec3 e1 = p1 - p0;
		glm::vec3 e2 = p2 - p0;

		float area = 0.5f * glm::length(glm::cross(e1, e2));

		float triMass = density * area;

		float share = triMass / 3.0f;
		masses[i0] += share;
		masses[i1] += share;
		masses[i2] += share;
	}

	// Set inverse masses using mass
	for (uint32_t i = cloth.offset_particle; i < cloth.offset_particle + cloth.num_particle; i++) {
		float m = masses[i];
		if (m > 0.0f)
			inverse_masses[i] = 1.0f / m;
		else
			inverse_masses[i] = 0.0f;
	}
}

void ParticleManager::CreateSSBO()
{
	// Self Collision
	uint32_t N = total_particles_;
	uint32_t tableSize = N;
	uint32_t maxNeighbors = 20;

	particle_hashes.resize(N, 0);
	particle_indices.resize(N, 0);

	starts.resize(tableSize, 0);
	ends.resize(tableSize, 0);

	num_neighbors = N * maxNeighbors;
	neighbors.resize(num_neighbors, 0);
	neighbor_lambdas.resize(num_neighbors, 0);

	// position
	ssbo_size_.position = sizeof(glm::vec4) * N;
	vku::CreateSSBO(context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.position,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		positions,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.position, ssbo_memories_.position,
		&staging_.position, &staging_memories_.position);
	staging_mapped_.position = staging_memories_.position.mapMemory(0, ssbo_size_.position);

	// Pred Position
	ssbo_size_.pred_position = sizeof(glm::vec4) * N;
	vku::CreateSSBO(context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.pred_position,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		pred_positions,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.pred_position, ssbo_memories_.pred_position,
		&staging_.pred_position, &staging_memories_.pred_position);
	staging_mapped_.pred_position = staging_memories_.pred_position.mapMemory(0, ssbo_size_.pred_position);

	// velocity
	ssbo_size_.velocity = sizeof(glm::vec4) * N;
	vku::CreateSSBO(context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.velocity,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		velocities,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.velocity, ssbo_memories_.velocity,
		&staging_.velocity, &staging_memories_.velocity);
	staging_mapped_.velocity = staging_memories_.velocity.mapMemory(0, ssbo_size_.velocity);

	// inverse mass
	ssbo_size_.inverse_mass = sizeof(float) * N;
	vku::CreateSSBO(context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.inverse_mass,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		inverse_masses,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.inverse_mass, ssbo_memories_.inverse_mass,
		&staging_.inverse_mass, &staging_memories_.inverse_mass);
	staging_mapped_.inverse_mass = staging_memories_.inverse_mass.mapMemory(0, ssbo_size_.inverse_mass);


	// particle_hash
	ssbo_size_.particle_hash = sizeof(uint32_t) * N;
	vku::CreateSSBO(context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.particle_hash,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		particle_hashes,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.particle_hash, ssbo_memories_.particle_hash);

	// particle_indice
	ssbo_size_.particle_indice = sizeof(uint32_t) * N;
	vku::CreateSSBO(context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.particle_indice,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		particle_indices,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.particle_indice, ssbo_memories_.particle_indice);

	// start
	ssbo_size_.start = sizeof(uint32_t) * tableSize;
	vku::CreateSSBO(context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.start,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		starts,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.start, ssbo_memories_.start);

	// end
	ssbo_size_.end = sizeof(uint32_t) * tableSize;
	vku::CreateSSBO(context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.end,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		ends,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.end, ssbo_memories_.end);

	// neighbor
	ssbo_size_.neighbor = sizeof(uint32_t) * num_neighbors;
	vku::CreateSSBO(context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.neighbor,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		neighbors,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.neighbor, ssbo_memories_.neighbor);

	// neighbor_lambda
	ssbo_size_.neighbor_lambda = sizeof(float) * num_neighbors;
	vku::CreateSSBO(context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.neighbor_lambda,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		neighbor_lambdas,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.neighbor_lambda, ssbo_memories_.neighbor_lambda);

}