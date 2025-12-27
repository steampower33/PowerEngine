
#include "context.h"
#include "vulkan_utils.h"
#include "model.h"
#include "model_manager.h"
#include "texture.h"
#include "texture_manager.h"
#include "geometry_generator.h"
#include "model_loader.h"

#include "particle_manager.h"

ParticleManager::ParticleManager(Context& context, ModelManager& modelManager, TextureManager& textureManager)
	: context_(context)
{
	{
		Cloth cloth{};

		cloth.name = "2x2 Cloth";
		cloth.spacing = default_cloth_spacing_;
		cloth.gsm = 0.2f;
		cloth.cloth_size = glm::vec2(2.0f, 2.0f);

		cloth.color = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);

		cloth.origin = glm::vec3(0.0f, 4.0f, 0.0f);
		cloth.angle_deg = 0.0f;
		cloth.axis = glm::vec3(0, 1, 0);

		cloth.num_particle = cloth.nx1 * cloth.ny1;

		cloth.render = true;

		std::string base = "assets/fabric/gingham_check";
		cloth.ubo_data.albedo_enable = 1;
		cloth.ubo_data.albedo_idx = textureManager.CreateTexture(base, "diff", false, true);
		cloth.ubo_data.normal_enable = 1;
		cloth.ubo_data.normal_idx = textureManager.CreateTexture(base, "nor", false, true);
		cloth.ubo_data.arm_enable = 1;
		cloth.ubo_data.arm_idx = textureManager.CreateTexture(base, "arm", false, true);
		cloth.ubo_data.fuzz_factor = 0.1f;
		cloth.ubo_data.fuzz_roughness_factor = 1.0f;

		cloth.ubo_data.tile_uv = cloth.cloth_size;

		SetPlaneCloth(cloth);

	}

	{
		Cloth cloth{};

		cloth.name = "3x3 Cloth";
		cloth.spacing = default_cloth_spacing_;
		cloth.gsm = 0.2f;
		cloth.cloth_size = glm::vec2(3.0f, 3.0f);

		cloth.color = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);

		cloth.origin = glm::vec3(0.0f, 3.0f, 0.0f);
		cloth.angle_deg = 0.0f;
		cloth.axis = glm::vec3(0, 1, 0);

		cloth.num_particle = cloth.nx1 * cloth.ny1;

		cloth.render = true;

		std::string base = "assets/fabric/terry_cloth";
		cloth.ubo_data.albedo_enable = 1;
		cloth.ubo_data.albedo_idx = textureManager.CreateTexture(base, "diff", false, true);
		cloth.ubo_data.normal_enable = 1;
		cloth.ubo_data.normal_idx = textureManager.CreateTexture(base, "nor", false, true);
		cloth.ubo_data.arm_enable = 1;
		cloth.ubo_data.arm_idx = textureManager.CreateTexture(base, "arm", false, true);
		cloth.ubo_data.fuzz_factor = 0.1f;
		cloth.ubo_data.fuzz_roughness_factor = 1.0f;

		cloth.ubo_data.tile_uv = cloth.cloth_size;

		SetPlaneCloth(cloth);

	}

	{
		SoftBody softbody;
		softbody.name = "softbody";
		softbody.origin = glm::vec3(10.0f, 1.0f, 0.0f);
		softbody.render = true;

		SetSoftbody("assets/sphere.msh", softbody);
	}

	vku::CreateIndexBuffer(context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_, indices_, index_buffer_, index_buffer_memory_);

	total_particles_ = num_cloth_particles_ + num_softbody_particles_;
	assert(total_particles_ == positions_.size());
	total_indices_ = num_cloth_indices_ + num_softbody_indices_;
	assert(total_indices_ == indices_.size());

	total_tries_ = (total_indices_ / 3);

	BuildVertexAdjacency();

	CreateSSBO();
}

ParticleManager::~ParticleManager()
{

}

void ParticleManager::SetPlaneCloth(Cloth& cloth)
{
	cloth.type = Cloth::Type::PLANE;
	cloth.nx = (uint32_t)std::round(cloth.cloth_size.x / cloth.spacing);
	cloth.ny = (uint32_t)std::round(cloth.cloth_size.y / cloth.spacing);
	cloth.nx1 = cloth.nx + 1;
	cloth.ny1 = cloth.ny + 1;

	const int nx = cloth.nx;
	const int ny = cloth.ny;
	const int nx1 = cloth.nx1;
	const int ny1 = cloth.ny1;

	cloth.offset_particle = num_cloth_particles_;
	cloth.offset_indices = num_cloth_indices_;

	auto vid = [&](int x, int y) { return cloth.offset_particle + uint32_t(y * nx1 + x); };

	const uint32_t N = nx1 * ny1;

	float angle_rad = glm::radians(cloth.angle_deg);
	glm::mat4 R = glm::rotate(glm::mat4(1.0f), angle_rad, glm::normalize(cloth.axis));

	// Set positions, velocities, pred_positions
	for (int y = 0; y < ny1; ++y) {
		for (int x = 0; x < nx1; ++x) {
			float lx = (-0.5f * cloth.cloth_size.x) + x * cloth.spacing;
			float ly = 0.0f;
			float lz = (-0.5f * cloth.cloth_size.y) + y * cloth.spacing;

			glm::vec4 local(lx, ly, lz, 1.0f);

			glm::vec3 rotated = glm::vec3(R * local);

			glm::vec3 worldPos = cloth.origin + rotated;

			positions_.push_back(glm::vec4(worldPos, 1.0f));
			velocities_.push_back(glm::vec4(0));
			pred_positions_.push_back(glm::vec4(worldPos, 1.0f));
			masses_.push_back(0.0f);
			inverse_masses_.push_back(0.0f);
			normals_.push_back(glm::vec4(0.0f));
			collision_masks_.push_back({ object_cnt_, ObjectType::CLOTH });
		}
	}
	cloth.num_particle = positions_.size() - cloth.offset_particle;

	// Set indices
	for (int y = 0; y < ny; ++y) {
		for (int x = 0; x < nx; ++x) {
			uint32_t i0 = vid(x, y);
			uint32_t i1 = vid(x + 1, y);
			uint32_t i2 = vid(x, y + 1);
			uint32_t i3 = vid(x + 1, y + 1);
			indices_.push_back(i0); indices_.push_back(i2); indices_.push_back(i1);
			indices_.push_back(i1); indices_.push_back(i2); indices_.push_back(i3);
		}
	}
	cloth.num_indices = static_cast<uint32_t>(indices_.size() - cloth.offset_indices);

	// Total area
	float totalArea = 0.0f;
	for (size_t t = cloth.offset_indices; t < cloth.offset_indices + cloth.num_indices; t += 3) {
		uint32_t i0 = indices_[t + 0];
		uint32_t i1 = indices_[t + 1];
		uint32_t i2 = indices_[t + 2];

		glm::vec3 p0 = glm::vec3(positions_[i0]);
		glm::vec3 p1 = glm::vec3(positions_[i1]);
		glm::vec3 p2 = glm::vec3(positions_[i2]);

		glm::vec3 e1 = p1 - p0;
		glm::vec3 e2 = p2 - p0;

		float area = 0.5f * glm::length(glm::cross(e1, e2)); // Triangle area
		totalArea += area;
	}

	float totalMassTarget = cloth.cloth_size.x * cloth.cloth_size.y * cloth.gsm;

	// Area zero defence
	float density = 0.0f;
	if (totalArea > 0.0f) {
		density = totalMassTarget / totalArea; // kg/m��
	}

	// Distribute mass to each triangle in proportion to area
	for (size_t t = cloth.offset_indices; t < cloth.offset_indices + cloth.num_indices; t += 3) {
		uint32_t i0 = indices_[t + 0];
		uint32_t i1 = indices_[t + 1];
		uint32_t i2 = indices_[t + 2];

		glm::vec3 p0 = glm::vec3(positions_[i0]);
		glm::vec3 p1 = glm::vec3(positions_[i1]);
		glm::vec3 p2 = glm::vec3(positions_[i2]);

		glm::vec3 e1 = p1 - p0;
		glm::vec3 e2 = p2 - p0;

		float area = 0.5f * glm::length(glm::cross(e1, e2));

		float triMass = density * area;

		float share = triMass / 3.0f;
		masses_[i0] += share;
		masses_[i1] += share;
		masses_[i2] += share;
	}

	// Set inverse masses using mass
	for (uint32_t i = cloth.offset_particle; i < cloth.offset_particle + cloth.num_particle; i++) {
		float m = masses_[i];
		if (m > 0.0f)
			inverse_masses_[i] = 1.0f / m;
		else
			inverse_masses_[i] = 0.0f;
	}

	num_cloth_particles_ = positions_.size();
	num_cloth_indices_ = indices_.size();

	object_cnt_++;

	clothes_.push_back(cloth);
}

void ParticleManager::SetClothFromMesh(Cloth& cloth)
{
	cloth.type = Cloth::Type::MESH;

	cloth.offset_particle = num_cloth_particles_;
	cloth.offset_indices = num_cloth_indices_;

	float angle_rad = glm::radians(cloth.angle_deg);
	glm::mat4 R = glm::rotate(glm::mat4(1.0f), angle_rad, glm::normalize(cloth.axis));

	// Set positions, velocities, pred_positions
	for (uint32_t i = 0; i < cloth.vertices.size(); i++)
	{
		glm::vec4 local(cloth.vertices[i].pos, 1.0f);

		glm::vec3 rotated = glm::vec3(R * local);

		glm::vec3 worldPos = cloth.origin + rotated;

		positions_.push_back(glm::vec4(worldPos, 1.0f));
		velocities_.push_back(glm::vec4(0));
		pred_positions_.push_back(glm::vec4(worldPos, 1.0f));
		masses_.push_back(0.0f);
		inverse_masses_.push_back(0.0f);
		normals_.push_back(glm::vec4(0.0f));
		collision_masks_.push_back({ object_cnt_, ObjectType::CLOTH });
	}

	cloth.num_particle = positions_.size() - cloth.offset_particle;

	// Set indices
	for (uint32_t i = 0; i < cloth.indices.size(); i++)
	{
		indices_.push_back(cloth.offset_particle + cloth.indices[i]);
	}
	cloth.num_indices = static_cast<uint32_t>(indices_.size() - cloth.offset_indices);

	// Total area
	float totalArea = 0.0f;
	for (size_t t = cloth.offset_indices; t < cloth.offset_indices + cloth.num_indices; t += 3) {
		uint32_t i0 = indices_[t + 0];
		uint32_t i1 = indices_[t + 1];
		uint32_t i2 = indices_[t + 2];

		glm::vec3 p0 = glm::vec3(positions_[i0]);
		glm::vec3 p1 = glm::vec3(positions_[i1]);
		glm::vec3 p2 = glm::vec3(positions_[i2]);

		glm::vec3 e1 = p1 - p0;
		glm::vec3 e2 = p2 - p0;

		float area = 0.5f * glm::length(glm::cross(e1, e2)); // Triangle area
		totalArea += area;
	}

	float totalMassTarget = 10.0f;

	// Area zero defence
	float density = 0.0f;
	if (totalArea > 0.0f) {
		density = totalMassTarget / totalArea; // kg/m��
	}

	// Distribute mass to each triangle in proportion to area
	for (size_t t = cloth.offset_indices; t < cloth.offset_indices + cloth.num_indices; t += 3) {
		uint32_t i0 = indices_[t + 0];
		uint32_t i1 = indices_[t + 1];
		uint32_t i2 = indices_[t + 2];

		glm::vec3 p0 = glm::vec3(positions_[i0]);
		glm::vec3 p1 = glm::vec3(positions_[i1]);
		glm::vec3 p2 = glm::vec3(positions_[i2]);

		glm::vec3 e1 = p1 - p0;
		glm::vec3 e2 = p2 - p0;

		float area = 0.5f * glm::length(glm::cross(e1, e2));

		float triMass = density * area;

		float share = triMass / 3.0f;
		masses_[i0] += share;
		masses_[i1] += share;
		masses_[i2] += share;
	}

	// Set inverse masses using mass
	for (uint32_t i = cloth.offset_particle; i < cloth.offset_particle + cloth.num_particle; i++) {
		float m = masses_[i];
		if (m > 0.0f)
			inverse_masses_[i] = 1.0f / m;
		else
			inverse_masses_[i] = 0.0f;
	}

	num_cloth_particles_ = positions_.size();
	num_cloth_indices_ = indices_.size();

	object_cnt_++;

	clothes_.push_back(cloth);

	//cloth.mesh = std::move(mesh);
}

void ParticleManager::SetSoftbody(std::string path, SoftBody& softbody)
{
	softbody.tetmesh = vku::LoadGmshMsh2(path.c_str());
	softbody.density = 0.5f;
	softbody.color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);

	softbody.offset_particle = positions_.size();
	softbody.offset_indices = indices_.size();

	softbody.num_particle = softbody.tetmesh.positions.size();
	softbody.num_indices = softbody.tetmesh.surfaceIndices.size();

	for (auto& p : softbody.tetmesh.positions)
	{
		glm::vec4 pos = glm::vec4(p, 1.0f) + glm::vec4(softbody.origin, 0.0f);
		positions_.push_back(pos);
		velocities_.push_back(glm::vec4(0.0f));
		pred_positions_.push_back(pos);
		masses_.push_back(0.0f);
		inverse_masses_.push_back(0.0f);
		normals_.push_back(glm::vec4(0.0f));
		collision_masks_.push_back({ object_cnt_, ObjectType::SOFTBODY });
	}

	for (auto idx : softbody.tetmesh.surfaceIndices)
	{
		indices_.push_back(softbody.offset_particle + idx);
	}

	num_softbody_particles_ += softbody.num_particle;
	num_softbody_indices_ += softbody.num_indices;

	auto SignedVolume = [&](uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) {
		glm::vec3 x0 = softbody.tetmesh.positions[i0];
		glm::vec3 x1 = softbody.tetmesh.positions[i1];
		glm::vec3 x2 = softbody.tetmesh.positions[i2];
		glm::vec3 x3 = softbody.tetmesh.positions[i3];
		return glm::dot(glm::cross(x1 - x0, x2 - x0), x3 - x0) / 6.0f;
		};

	for (auto t : softbody.tetmesh.tets)
	{
		float v = SignedVolume(t.x, t.y, t.z, t.w);

		if (std::abs(v) < 1e-12f) continue;

		if (v < 0.0f) {
			std::swap(t.y, t.z);
			v = -v;
		}

		Volume c;
		c.i0 = softbody.offset_particle + t.x;
		c.i1 = softbody.offset_particle + t.y;
		c.i2 = softbody.offset_particle + t.z;
		c.i3 = softbody.offset_particle + t.w;

		c.rest_volume = v;
		c.lambda = 0.0f;

		volume_constraints.push_back(c);

		float tetMass = softbody.density * c.rest_volume;
		float pMass = tetMass * 0.25f;

		masses_[c.i0] += pMass;
		masses_[c.i1] += pMass;
		masses_[c.i2] += pMass;
		masses_[c.i3] += pMass;
	}

	for (uint32_t i = softbody.offset_particle; i < softbody.offset_particle + softbody.num_particle; ++i)
	{
		inverse_masses_[i] = (masses_[i] > 0.0f) ? (1.0f / masses_[i]) : 0.0f;
	}

	object_cnt_++;

	softbodies_.push_back(softbody);
}

void ParticleManager::ResetCloth(Cloth& cloth)
{
	if (cloth.type == Cloth::Type::MESH)
		return;

	const int nxCells = cloth.nx;
	const int nyCells = cloth.ny;
	const int nx1 = nxCells + 1;
	const int ny1 = nyCells + 1;

	auto vid = [&](int x, int y) { return cloth.offset_particle + uint32_t(y * nx1 + x); };

	const uint32_t N = nx1 * ny1;

	float angle_rad = glm::radians(cloth.angle_deg);
	glm::mat4 R = glm::rotate(glm::mat4(1.0f), angle_rad, glm::normalize(cloth.axis));

	// Set positions, velocities, pred_positions

	if (cloth.type == Cloth::Type::MESH)
	{

	}
	else if (cloth.type == Cloth::Type::PLANE)
	{
		for (int y = 0; y < ny1; ++y) {
			for (int x = 0; x < nx1; ++x) {
				uint32_t id = vid(x, y);

				float lx = (-0.5f * cloth.cloth_size.x) + x * cloth.spacing;
				float ly = 0.0f;
				float lz = (-0.5f * cloth.cloth_size.y) + y * cloth.spacing;

				glm::vec4 local(lx, ly, lz, 1.0f);

				glm::vec3 rotated = glm::vec3(R * local);

				glm::vec3 worldPos = cloth.origin + rotated;

				positions_[id] = { worldPos, 1.0f };
				velocities_[id] = glm::vec4(0.0f);
				pred_positions_[id] = { worldPos, 1.0f };
			}
		}
	}

	// Total area
	float totalArea = 0.0f;
	for (size_t t = cloth.offset_indices; t < cloth.offset_indices + cloth.num_indices; t += 3) {
		uint32_t i0 = indices_[t + 0];
		uint32_t i1 = indices_[t + 1];
		uint32_t i2 = indices_[t + 2];

		glm::vec3 p0 = glm::vec3(positions_[i0]);
		glm::vec3 p1 = glm::vec3(positions_[i1]);
		glm::vec3 p2 = glm::vec3(positions_[i2]);

		glm::vec3 e1 = p1 - p0;
		glm::vec3 e2 = p2 - p0;

		float area = 0.5f * glm::length(glm::cross(e1, e2)); // Triangle area
		totalArea += area;
	}

	float totalMassTarget = cloth.cloth_size.x * cloth.cloth_size.y * cloth.gsm;

	// Area zero defence
	float density = 0.0f;
	if (totalArea > 0.0f) {
		density = totalMassTarget / totalArea; // kg/m��
	}

	for (uint32_t i = cloth.offset_particle; i < cloth.offset_particle + cloth.num_particle; ++i)
		masses_[i] = 0.0f;

	// Distribute mass to each triangle in proportion to area
	for (size_t t = cloth.offset_indices; t < cloth.offset_indices + cloth.num_indices; t += 3) {
		uint32_t i0 = indices_[t + 0];
		uint32_t i1 = indices_[t + 1];
		uint32_t i2 = indices_[t + 2];

		glm::vec3 p0 = glm::vec3(positions_[i0]);
		glm::vec3 p1 = glm::vec3(positions_[i1]);
		glm::vec3 p2 = glm::vec3(positions_[i2]);

		glm::vec3 e1 = p1 - p0;
		glm::vec3 e2 = p2 - p0;

		float area = 0.5f * glm::length(glm::cross(e1, e2));

		float triMass = density * area;

		float share = triMass / 3.0f;
		masses_[i0] += share;
		masses_[i1] += share;
		masses_[i2] += share;
	}

	// Set inverse masses using mass
	for (uint32_t i = cloth.offset_particle; i < cloth.offset_particle + cloth.num_particle; i++) {
		float m = masses_[i];
		if (m > 0.0f)
			inverse_masses_[i] = 1.0f / m;
		else
			inverse_masses_[i] = 0.0f;
	}
}

void ParticleManager::ResetSoftbody(SoftBody& softbody)
{
	for (uint32_t i = 0; i < softbody.tetmesh.positions.size(); i++)
	{
		glm::vec4 pos = glm::vec4(softbody.tetmesh.positions[i] + softbody.origin, 1.0f);
		positions_[softbody.offset_particle + i] = pos;
		pred_positions_[softbody.offset_particle + i] = pos;
	}
}

void ParticleManager::ResetVolumeConstraint()
{
	for (auto& v : volume_constraints)
	{
		v.lambda = 0.0f;
	}
}

void ParticleManager::CreateSSBO()
{
	// Self Collision
	uint32_t tableSize = total_particles_;
	uint32_t maxNeighbors = 16;

	particle_hashes_.resize(total_particles_, 0);
	particle_indices_.resize(total_particles_, 0);

	starts_.resize(tableSize, 0);
	ends_.resize(tableSize, 0);

	num_neighbors_ = total_particles_ * maxNeighbors;
	neighbors_.resize(num_neighbors_, 0);
	neighbor_lambdas_.resize(num_neighbors_, 0);

	// position
	ssbo_size_.position = sizeof(glm::vec4) * total_particles_;
	vku::CreateSSBO("Position", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.position,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		positions_,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.position, ssbo_memories_.position,
		&staging_.position, &staging_memories_.position);
	staging_mapped_.position = staging_memories_.position.mapMemory(0, ssbo_size_.position);

	// Pred Position
	ssbo_size_.pred_position = sizeof(glm::vec4) * total_particles_;
	vku::CreateSSBO("Pred Position", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.pred_position,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		pred_positions_,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.pred_position, ssbo_memories_.pred_position,
		&staging_.pred_position, &staging_memories_.pred_position);
	staging_mapped_.pred_position = staging_memories_.pred_position.mapMemory(0, ssbo_size_.pred_position);

	// velocity
	ssbo_size_.velocity = sizeof(glm::vec4) * total_particles_;
	vku::CreateSSBO("Velocity", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.velocity,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		velocities_,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.velocity, ssbo_memories_.velocity,
		&staging_.velocity, &staging_memories_.velocity);
	staging_mapped_.velocity = staging_memories_.velocity.mapMemory(0, ssbo_size_.velocity);

	// inverse mass
	ssbo_size_.inverse_mass = sizeof(float) * total_particles_;
	vku::CreateSSBO("Inverse Mass", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.inverse_mass,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		inverse_masses_,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.inverse_mass, ssbo_memories_.inverse_mass,
		&staging_.inverse_mass, &staging_memories_.inverse_mass);
	staging_mapped_.inverse_mass = staging_memories_.inverse_mass.mapMemory(0, ssbo_size_.inverse_mass);

	// particle_hash
	ssbo_size_.particle_hash = sizeof(uint32_t) * total_particles_;
	vku::CreateSSBO("Particle Hash", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.particle_hash,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		particle_hashes_,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.particle_hash, ssbo_memories_.particle_hash);

	// particle_indice
	ssbo_size_.sorted_indice = sizeof(uint32_t) * total_particles_;
	vku::CreateSSBO("Particle Indice", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.sorted_indice,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		particle_indices_,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.sorted_indice, ssbo_memories_.sorted_indice);

	// start
	ssbo_size_.start = sizeof(uint32_t) * tableSize;
	vku::CreateSSBO("Start", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.start,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		starts_,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.start, ssbo_memories_.start);

	// end
	ssbo_size_.end = sizeof(uint32_t) * tableSize;
	vku::CreateSSBO("End", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.end,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		ends_,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.end, ssbo_memories_.end);

	// neighbor
	ssbo_size_.neighbor = sizeof(uint32_t) * num_neighbors_;
	vku::CreateSSBO("Neighbor", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.neighbor,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		neighbors_,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.neighbor, ssbo_memories_.neighbor);

	// neighbor_lambda
	ssbo_size_.neighbor_lambda = sizeof(float) * num_neighbors_;
	vku::CreateSSBO("Neighbor Lambda", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.neighbor_lambda,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		neighbor_lambdas_,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.neighbor_lambda, ssbo_memories_.neighbor_lambda);

	// index
	ssbo_size_.index = sizeof(uint32_t) * total_indices_;
	vku::CreateSSBO("Index", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.index,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		indices_,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.index, ssbo_memories_.index);

	// normals
	ssbo_size_.normal = sizeof(glm::vec4) * total_particles_;
	vku::CreateSSBO("Normal", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.normal,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		normals_,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.normal, ssbo_memories_.normal);

	// tri_normals
	tri_normals_.resize(total_tries_, glm::vec4(0.0f));
	ssbo_size_.tri_normals = sizeof(glm::vec4) * total_tries_;
	vku::CreateSSBO("Tri Normal", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.tri_normals,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		tri_normals_,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.tri_normals, ssbo_memories_.tri_normals);

	// vertex_tri_offsets
	ssbo_size_.vertex_tri_offsets = sizeof(uint32_t) * vertex_tri_offsets_.size();
	vku::CreateSSBO("Vertex Tri Offset", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.vertex_tri_offsets,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		vertex_tri_offsets_,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.vertex_tri_offsets, ssbo_memories_.vertex_tri_offsets);

	// vertex_tri_indices
	ssbo_size_.vertex_tri_indices = sizeof(uint32_t) * vertex_tri_indices_.size();
	vku::CreateSSBO("Vertex Tri Indices", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.vertex_tri_indices,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		vertex_tri_indices_,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.vertex_tri_indices, ssbo_memories_.vertex_tri_indices);

	// object_ids_
	ssbo_size_.collision_masks_ = sizeof(ColiisiotnMask) * collision_masks_.size();
	vku::CreateSSBO("Object Id", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		ssbo_size_.collision_masks_,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		collision_masks_,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.collision_masks_, ssbo_memories_.collision_masks_);

}

void ParticleManager::BuildVertexAdjacency()
{
	const uint32_t numIndices = total_indices_;
	assert(numIndices % 3 == 0);
	const uint32_t numTris = total_tries_;

	uint32_t numVertices = total_particles_;
	vertex_tri_offsets_.assign(numVertices + 1, 0);
	vertex_tri_indices_.resize(numIndices);

	for (uint32_t tri = 0; tri < numTris; ++tri)
	{
		uint32_t i0 = indices_[tri * 3 + 0];
		uint32_t i1 = indices_[tri * 3 + 1];
		uint32_t i2 = indices_[tri * 3 + 2];

		assert(i0 < numVertices && i1 < numVertices && i2 < numVertices);

		vertex_tri_offsets_[i0]++;
		vertex_tri_offsets_[i1]++;
		vertex_tri_offsets_[i2]++;
	}

	uint32_t sum = 0;
	for (uint32_t v = 0; v < numVertices; ++v)
	{
		uint32_t count = vertex_tri_offsets_[v];
		vertex_tri_offsets_[v] = sum;
		sum += count;
	}
	vertex_tri_offsets_[numVertices] = sum;

	std::vector<uint32_t> cursor = vertex_tri_offsets_;

	for (uint32_t tri = 0; tri < numTris; ++tri)
	{
		uint32_t i0 = indices_[tri * 3 + 0];
		uint32_t i1 = indices_[tri * 3 + 1];
		uint32_t i2 = indices_[tri * 3 + 2];

		vertex_tri_indices_[cursor[i0]++] = tri;
		vertex_tri_indices_[cursor[i1]++] = tri;
		vertex_tri_indices_[cursor[i2]++] = tri;
	}
}