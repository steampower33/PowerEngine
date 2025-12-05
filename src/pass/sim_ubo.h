#pragma once

struct SimUBO {
	struct Data {
		struct SimParams {
			glm::vec4 gravity = glm::vec4(0.0f, -9.8f, 0.0f, 0.0f);
			glm::vec4 sphere_center;
			float sphere_radius;
			float thickness = 0.004f;
			float friction = 1.0f;
			float dt = 0.0f;
			float global_damping = 2.0f;
			float relaxation_factor = 0.2f;
			float neighbor_friction = 1.0f;
			uint32_t num_particles;
			uint32_t num_edges;
			uint32_t num_shears;
			uint32_t num_bends;
			uint32_t num_areas;

			float cell_size;
			uint32_t num_tables;
			uint32_t max_neighbors;
			float collision_radius;

			float stretch_stiffness = 1.0f;
			float shear_stiffness = 10.0f;
			float bend_stiffness = 0.01f;
			float area_stiffness = 1.0f;

			float self_collision_stiffness = 25.0f;
			float max_speed;
			uint32_t num_tries;
			float p3;
		} sim_params;
		static_assert(sizeof(SimParams) % 16 == 0, "std140 must be 16-byte aligned.");
	} datas;

	struct UBO {
		vk::raii::Buffer sim_params{ nullptr };
	} ubos;

	struct Memory {
		vk::raii::DeviceMemory sim_params{ nullptr };
	} memories;

	struct Mapped {
		void* sim_params{ nullptr };
	} mapped;

	struct UBOSize {
		vk::DeviceSize sim_params;
	} size;

};