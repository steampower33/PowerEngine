#pragma once

struct SimUBO {
	struct Data {
		struct SimParams {
			glm::vec4 gravity = glm::vec4(0.0f, -9.8f, 0.0f, 0.0f);
			glm::vec4 sphere_center;
			float sphere_radius;
			float thickness = 0.05f;
			float friction = 0.1f;
			float dt = 0.0f;
			float global_damping = 0.25f;
			float relaxation_factor = 0.2f;
			uint32_t num_particles;
			uint32_t num_edges;
			uint32_t num_shears;
			uint32_t num_bends;
			uint32_t num_areas;
			uint32_t p0;

			float cell_size;
			uint32_t num_tables;
			uint32_t max_neighbors;
			float collision_radius;

			float shear_stiffness = 1.0f;
			float bend_stiffness = 1.0f;
			float area_stiffness = 1.0f;
			float self_collision_stiffness = 20.0f;
		} sim_params;
		static_assert(sizeof(SimParams) % 16 == 0, "std140 must be 16-byte aligned.");

		struct Render {
			glm::vec4 albedo_use{ 1.0f, 1.0f, 1.0f, 0.0f };

			int albedo_idx = -1;
			int metallic_idx = -1;
			int normal_idx = -1;
			int roughness_idx = -1;

			int ao_idx = -1;
			int height_idx = -1;
			float metallic_factor = 0.0f;
			float roughness_factor = 1.0f;

			float ao_factor = 1.0f;
			float height_factor = 0.0f;
			uint32_t p0 = 0;
			uint32_t p1 = 0;

			uint32_t albedo_enable = 0;
			uint32_t metallic_enable = 0;
			uint32_t normal_enable = 0;
			uint32_t roughtnessEnable = 0;

			uint32_t ao_enable = 0;
			uint32_t height_enable = 0;
			uint32_t p3;
			uint32_t p4;
		} render;
		static_assert(sizeof(Render) % 16 == 0, "std140 must be 16-byte aligned.");

	} datas;

	struct UBO {
		vk::raii::Buffer sim_params{ nullptr };
		vk::raii::Buffer render{ nullptr };
	} ubos;

	struct Memory {
		vk::raii::DeviceMemory sim_params{ nullptr };
		vk::raii::DeviceMemory render{ nullptr };
	} memories;

	struct Mapped {
		void* sim_params{ nullptr };
		void* render{ nullptr };
	} mapped;

	struct UBOSize {
		vk::DeviceSize sim_params;
		vk::DeviceSize render;
	} size;

};