#pragma once

namespace ubo_data {

	struct Global {
		glm::mat4 view;
		glm::mat4 proj;
	};
	static_assert(sizeof(Global) % 16 == 0, "std140 must be 16-byte aligned.");

	struct Light {
		glm::mat4 invViewProj{};
		glm::vec4 cameraPos{};

		glm::vec3 position{ 0.0f, 10.0f, 5.0f };
		float intensity = 200.0f;

		glm::vec3 direction{ 0.0f, -1.0f, -1.0f };
		float inner = 0.0f;

		float outer = 90.0f;
		uint32_t light_enable = 1;
		uint32_t pbr_enable = 0;
		float exposure = 0.8f;

		int ggx_brdf_idx = 0;
		int charlie_brdf_idx = 0;
		int sheen_e_brdf_idx = 0;
		int p0;
	};
	static_assert(sizeof(Light) % 16 == 0, "std140 must be 16-byte aligned.");

	struct SkyBox {
		int env_idx = 0;
		int specular_idx = 0;
		int diffuse_idx = 0;
		uint32_t specular_mip_levels = 0;
	};
	static_assert(sizeof(SkyBox) % 16 == 0, "std140 must be 16-byte aligned.");

	struct Model {
		glm::mat4 world;
		glm::vec4 albedo{ 1.0f, 1.0f, 1.0f, 0.0f };

		int albedo_idx = -1;
		int normal_idx = -1;
		int arm_idx = -1;
		int ao_idx = -1;

		int roughness_idx = -1;
		int metallic_idx = -1;
		int height_idx = -1;
		int coat_idx = -1;

		int fuzz_idx = -1;
		float ao_factor = 1.0f;
		float roughness_factor = 1.0f;
		float metallic_factor = 0.0f;

		float height_factor = 0.0f;
		float coat_factor = 0.0f;
		float coat_roughness_factor = 0.0f;
		float fuzz_factor = 0.0f;

		float fuzz_roughness_factor = 0.0f;
		uint32_t albedo_enable = 0;
		uint32_t normal_enable = 0;
		uint32_t arm_enable = 0;

		uint32_t ao_enable = 0;
		uint32_t roughness_enable = 0;
		uint32_t metallic_enable = 0;
		uint32_t height_enable = 0;

		uint32_t coat_enable = 0;
		uint32_t fuzz_enable = 0;
		uint32_t checker_board_enable = 0;
		uint32_t p0;

		glm::vec2 tile_uv{ 1.0f, 1.0f };
		float p1;
		float p2;
	};
	static_assert(sizeof(Model) % 16 == 0, "std140 must be 16-byte aligned.");
}