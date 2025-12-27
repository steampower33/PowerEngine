#pragma once

struct SimUBO {
	struct Data {
		struct SimParams {
			glm::vec4 gravity = glm::vec4(0.0f, -9.8f, 0.0f, 0.0f);

			float dt = 0.0f;
			float thickness = 0.004f;
			float friction = 1.0f;
			float max_speed;
			
			float global_damping = 2.0f;
			float relaxation_factor = 0.2f;
			float neighbor_friction = 0.1f;
			float p0;
			
			uint32_t num_particles;
			uint32_t num_edges;
			uint32_t num_shears;
			uint32_t num_bends;

			uint32_t num_areas;
			uint32_t num_tries;
			uint32_t num_volumes;
			uint32_t num_colliders;

			float cell_size;
			uint32_t num_tables;
			uint32_t max_neighbors;
			float collision_radius;

			float stretch_stiffness = 1.0f;
			float shear_stiffness = 5.0f;
			float bend_stiffness = 0.001f;
			float area_stiffness = 1.0f;

			float self_collision_stiffness = 30.0f;
			float volume_stiffness = 10.0f;
			float softbody_stretch_stiffness = 20.0f;
			float inter_collision_stiffness = 100.0f;

			glm::vec3 wind_dir{ 0.0f, 0.0f, 1.0f };
			uint32_t wind_enable = 0;

			float wind_force = 1.0f;
			float air_density = 1.2f;
			float drag_coefficient = 4.0f;
			float lift_coefficient = 5.0f;
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