#pragma once

#include "model_data.h"
#include "particle_manager.h"

struct SimData {

	struct Compliance {
		float stretch = 1e-6f;
		float softbody_stretch = 1e-6f;
		float softbody_volume = 1e-6f;
		float shear = 1e-6f;
		float bend = 1e-2f;
		float area = 1e-2f;
		float self_collision = 1e-9f;
	} compliance;

	struct Beta {
		float stretch = 100.0f;
	} beta;

	uint32_t num_edges = 0;
	uint32_t num_shears = 0;
	uint32_t num_bends = 0;
	uint32_t num_areas = 0;
	uint32_t num_volumes = 0;
	uint32_t num_colliders = 0;

	uint32_t num_cloth_edges = 0;
	uint32_t num_softbody_edges = 0;

	float frame_dt = 60.0f;
	int substeps = 10;
	int iterations = 4;

	struct Edge {
		uint32_t i;
		uint32_t j;
		float    rest;
		float    lambda;
	};
	static_assert(sizeof(Edge) == 16, "Edge must be 16 bytes");
	std::vector<Edge> edges;
	std::vector<uint32_t> pass_offsets;  // cloth + softbody

	uint32_t num_colors = 0;

	// SoftBody Stretch Edge
	std::vector<Edge> softbody_stretch_edges;

	struct Shear {
		uint32_t i0, i1, i2;
		float rest_dot;

		float lambda;
		float p0;
		float p1;
		float p2;
	};
	static_assert(sizeof(Shear) == 32, "Shear must be 32 bytes");
	std::vector<Shear> shears;

	struct Bend {
		uint32_t i0, i1, i2, i3;
		float rest_angle;
		float lambda;
		float p0;
		float p1;
	};
	static_assert(sizeof(Bend) == 32, "Bend must be 32 bytes");
	std::vector<Bend> bends;

	struct Area {
		uint32_t i0, i1, i2;
		float rest_area;
		glm::vec3 rest_normal;
		float lambda;
	};
	static_assert(sizeof(Area) == 32, "Area must be 32 bytes");
	std::vector<Area> areas;

	struct EdgeKey
	{
		uint32_t a, b; // always a < b

		bool operator==(const EdgeKey& o) const noexcept {
			return a == o.a && b == o.b;
		}
	};

	struct EdgeKeyHash
	{
		size_t operator()(const EdgeKey& k) const noexcept {
			return (size_t(k.a) << 32) ^ size_t(k.b);
		}
	};

	struct TriRef
	{
		uint32_t triIndex;
		uint32_t oppVertex;
	};

	std::vector<Collider> colliders;

	struct LRAEntry {
		uint32_t anchor = 0xFFFFFFFFu;
		float r = std::numeric_limits<float>::infinity();
	};
	std::vector<uint32_t> lra_ids;
	std::vector<float> lra_r;

	float ComputeRestBendAngle(
		uint32_t i1, uint32_t i2,
		uint32_t i3, uint32_t i4,
		const std::vector<glm::vec4>& pos)
	{
		glm::vec3 p1 = glm::vec3(pos[i1]);
		glm::vec3 p2 = glm::vec3(pos[i2]);
		glm::vec3 p3 = glm::vec3(pos[i3]);
		glm::vec3 p4 = glm::vec3(pos[i4]);

		glm::vec3 e = p2 - p1;
		float el = glm::length(e);
		if (el < 1e-8f) return 0.0f;
		glm::vec3 ehat = e / el;

		glm::vec3 n1 = glm::normalize(glm::cross(p2 - p1, p3 - p1));
		glm::vec3 n2 = glm::normalize(glm::cross(p2 - p1, p4 - p1));

		float c = glm::clamp(glm::dot(n1, n2), -1.0f, 1.0f);
		glm::vec3 cross_n1n2 = glm::cross(n1, n2);
		float s = glm::dot(ehat, cross_n1n2);

		float phi = std::atan2(s, c); // atan(s,c)

		return phi;
	}

	void BuildStretchConstraints(std::vector<glm::vec4>& positions, std::vector<uint32_t>& indices, std::vector<Cloth> cloth)
	{
		uint32_t numIndices = 0;
		for (auto& c : cloth)
		{
			numIndices += c.num_indices;
		}

		const size_t numTris = numIndices / 3;

		struct RawEdge {
			uint32_t i;
			uint32_t j;
		};
		std::vector<RawEdge> rawEdges;

		auto addEdge = [&](uint32_t a, uint32_t b)
			{
				if (a == b) return;
				if (a > b) std::swap(a, b);
				rawEdges.push_back({ a, b });
			};

		for (size_t t = 0; t < numTris; ++t)
		{
			uint32_t i0 = indices[3 * t + 0];
			uint32_t i1 = indices[3 * t + 1];
			uint32_t i2 = indices[3 * t + 2];

			addEdge(i0, i1);
			addEdge(i1, i2);
			addEdge(i2, i0);
		}

		std::sort(rawEdges.begin(), rawEdges.end(),
			[](const RawEdge& a, const RawEdge& b)
			{
				if (a.i != b.i) return a.i < b.i;
				return a.j < b.j;
			});

		rawEdges.erase(
			std::unique(rawEdges.begin(), rawEdges.end(),
				[](const RawEdge& a, const RawEdge& b)
				{
					return (a.i == b.i) && (a.j == b.j);
				}),
			rawEdges.end()
		);

		struct StretchEdge {
			uint32_t i;
			uint32_t j;
			float    rest;
			uint32_t color;
		};
		std::vector<StretchEdge> stretch;
		edges.reserve(rawEdges.size());

		for (auto& e : rawEdges)
		{
			StretchEdge se{};
			se.i = e.i;
			se.j = e.j;

			glm::vec3 pi = positions[se.i];
			glm::vec3 pj = positions[se.j];
			se.rest = glm::length(pj - pi);
			if (se.rest < 1e-6f) continue;

			stretch.push_back(se);
		}

		std::vector<uint32_t> vertMask(positions.size(), 0u);

		uint32_t maxColorUsed = 0;

		for (auto& e : stretch)
		{
			uint32_t im = vertMask[e.i];
			uint32_t jm = vertMask[e.j];

			uint32_t used = im | jm;

			uint32_t color = 0;
			while (used & (1u << color)) {
				++color;
			}

			e.color = color;
			maxColorUsed = std::max(maxColorUsed, color);

			vertMask[e.i] |= (1u << color);
			vertMask[e.j] |= (1u << color);
		}

		//std::cout << "Stretch edges colored. numColors = " << (maxColorUsed + 1) << std::endl;

		std::sort(stretch.begin(), stretch.end(),
			[](const StretchEdge& a, const StretchEdge& b) {
				return a.color < b.color;
			});

		num_colors = maxColorUsed + 1;
		std::vector<uint32_t> colorOffset(num_colors + 1, 0);

		uint32_t currentColor = 0;
		colorOffset[0] = 0;
		for (uint32_t i = 0; i < stretch.size(); ++i) {
			StretchEdge s = stretch[i];
			while (currentColor < s.color) {
				++currentColor;
				colorOffset[currentColor] = i;
			}

			Edge e{};
			e.i = s.i;
			e.j = s.j;
			e.rest = s.rest;
			e.lambda = 0.0f;
			edges.push_back(e);
		}
		colorOffset[num_colors] = edges.size();

		pass_offsets = std::move(colorOffset);

		num_cloth_edges = static_cast<uint32_t>(edges.size());
	}

	void BuildShearConstraints(std::vector<glm::vec4>& positions, std::vector<uint32_t>& indices, std::vector<Cloth> cloth)
	{
		uint32_t numIndices = 0;
		for (auto& c : cloth)
		{
			numIndices += c.num_indices;
		}

		const size_t numTris = numIndices / 3;

		for (size_t t = 0; t < numTris; ++t)
		{
			uint32_t i0 = indices[3 * t + 0];
			uint32_t i1 = indices[3 * t + 1];
			uint32_t i2 = indices[3 * t + 2];

			const glm::vec3& x0 = positions[i0];
			const glm::vec3& x1 = positions[i1];
			const glm::vec3& x2 = positions[i2];

			glm::vec3 e1 = x1 - x0;
			glm::vec3 e2 = x2 - x0;

			float restDot = glm::dot(e1, e2);

			SimData::Shear c{};
			c.i0 = i0;
			c.i1 = i1;
			c.i2 = i2;
			c.rest_dot = restDot;
			c.lambda = 0.0f;

			shears.push_back(c);
		}
		num_shears = static_cast<uint32_t>(shears.size());
	}

	void BuildBendConstraints(std::vector<glm::vec4>& positions, std::vector<uint32_t>& indices, std::vector<Cloth> cloth)
	{
		bends.clear();

		std::unordered_map<EdgeKey, std::pair<TriRef, TriRef>, EdgeKeyHash> edgeMap;
		edgeMap.reserve(indices.size());

		auto make_edge = [](uint32_t i, uint32_t j) {
			EdgeKey k;
			if (i < j) { k.a = i; k.b = j; }
			else { k.a = j; k.b = i; }
			return k;
			};

		uint32_t numIndices = 0;
		for (auto& c : cloth)
		{
			numIndices += c.num_indices;
		}

		const size_t numTris = numIndices / 3;

		for (size_t t = 0; t < numTris; ++t)
		{
			uint32_t i0 = indices[3 * t + 0];
			uint32_t i1 = indices[3 * t + 1];
			uint32_t i2 = indices[3 * t + 2];

			// tri edges: (i0,i1), (i1,i2), (i2,i0)
			EdgeKey e01 = make_edge(i0, i1);
			EdgeKey e12 = make_edge(i1, i2);
			EdgeKey e20 = make_edge(i2, i0);

			TriRef r0{ t, i2 };
			TriRef r1{ t, i0 };
			TriRef r2{ t, i1 };

			auto insert_ref = [&](const EdgeKey& e, const TriRef& r)
				{
					auto it = edgeMap.find(e);
					if (it == edgeMap.end())
					{
						edgeMap.emplace(e, std::make_pair(r, TriRef{ UINT32_MAX, UINT32_MAX }));
					}
					else
					{
						if (it->second.second.triIndex == UINT32_MAX)
							it->second.second = r;
					}
				};

			insert_ref(e01, r0);
			insert_ref(e12, r1);
			insert_ref(e20, r2);
		}

		bends.reserve(edgeMap.size());

		for (const auto& kv : edgeMap)
		{
			const EdgeKey& e = kv.first;
			const TriRef& t0 = kv.second.first;
			const TriRef& t1 = kv.second.second;

			if (t1.triIndex == UINT32_MAX)
				continue;

			uint32_t i1 = e.a;         // hinge start
			uint32_t i2 = e.b;         // hinge end
			uint32_t i3 = t0.oppVertex;// tri0 opposite
			uint32_t i4 = t1.oppVertex;// tri1 opposite

			float restAngle = ComputeRestBendAngle(i1, i2, i3, i4, positions);

			Bend bc{};
			bc.i0 = i1;
			bc.i1 = i2;
			bc.i2 = i3;
			bc.i3 = i4;
			bc.rest_angle = restAngle;
			bc.lambda = 0.0f;

			bends.push_back(bc);
		}

		num_bends = static_cast<uint32_t>(bends.size());
	}

	void BuildAreaConstraints(std::vector<glm::vec4>& positions, std::vector<uint32_t>& indices, std::vector<Cloth> cloth)
	{
		uint32_t numIndices = 0;
		for (auto& c : cloth)
		{
			numIndices += c.num_indices;
		}

		const size_t numTris = numIndices / 3;

		for (size_t t = 0; t < numTris; ++t)
		{
			uint32_t i0 = indices[3 * t + 0];
			uint32_t i1 = indices[3 * t + 1];
			uint32_t i2 = indices[3 * t + 2];

			glm::vec3 p0 = positions[i0];
			glm::vec3 p1 = positions[i1];
			glm::vec3 p2 = positions[i2];

			glm::vec3 e0 = p1 - p0;
			glm::vec3 e1 = p2 - p0;

			glm::vec3 restNormal = glm::cross(e0, e1);
			float restArea = 0.5f * glm::length(restNormal);

			glm::vec3 nHat = (restArea > 0.0f) ? (restNormal / (2.0f * restArea)) : glm::vec3(0, 1, 0);

			Area area{};
			area.i0 = i0;
			area.i1 = i1;
			area.i2 = i2;
			area.lambda = 0.0f;
			area.rest_area = restArea;
			area.rest_normal = nHat;

			areas.push_back(area);
		}

		num_areas = static_cast<uint32_t>(areas.size());
	}

	void InsertBestK(std::vector<LRAEntry>& best, uint32_t a, float r) {
		if (!std::isfinite(r)) return;
		for (auto& e : best) {
			if (e.anchor == a) { e.r = std::min(e.r, r); return; }
		}
		int worst = 0;
		for (int k = 1; k < (int)best.size(); ++k) {
			if (best[k].r > best[worst].r) worst = k;
		}
		if (r < best[worst].r) best[worst] = { a, r };
	}

	std::vector<float> Dijkstra(
		int localVCount,
		const std::vector<std::vector<std::pair<int, float>>>& adj,
		int src)
	{
		const float INF = std::numeric_limits<float>::infinity();
		std::vector<float> dist(localVCount, INF);
		dist[src] = 0.0f;

		using Node = std::pair<float, int>; // (dist, v)
		std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
		pq.push({ 0.0f, src });

		while (!pq.empty()) {
			auto [d, u] = pq.top(); pq.pop();
			if (d != dist[u]) continue;
			for (auto [v, w] : adj[u]) {
				float nd = d + w;
				if (nd < dist[v]) {
					dist[v] = nd;
					pq.push({ nd, v });
				}
			}
		}
		return dist;
	}

	void BuildLRAConstraints(std::vector<glm::vec4>& positions, std::vector<uint32_t>& indices, ParticleManager& pm)
	{
		uint32_t K = 2;

		lra_ids.resize(pm.num_cloth_particles_ * K);
		lra_r.resize(pm.num_cloth_particles_ * K);
		for (auto& cloth : pm.clothes_)
		{
			std::vector<uint32_t> verts;
			uint32_t offsetParticles = cloth.offset_particle;
			uint32_t numParticles = cloth.num_particle;
			verts.reserve(numParticles);
			std::unordered_map<uint32_t, int> globalToLocal;
			globalToLocal.reserve(numParticles);

			for (uint32_t i = offsetParticles; i < offsetParticles + numParticles; ++i) {
				int li = (int)verts.size();
				verts.push_back(i);
				globalToLocal[i] = li;
			}
			const int V = (int)verts.size();
			if (V == 0) return;

			std::vector<std::vector<std::pair<int, float>>> adj(V);
			for (const auto& e : edges) {
				auto itI = globalToLocal.find(e.i);
				auto itJ = globalToLocal.find(e.j);
				if (itI == globalToLocal.end() || itJ == globalToLocal.end()) continue;

				int u = itI->second;
				int v = itJ->second;
				float w = e.rest;

				adj[u].push_back({ v, w });
				adj[v].push_back({ u, w });
			}

			std::vector<uint32_t> anchors;
			anchors.push_back(cloth.offset_particle);
			anchors.push_back(cloth.offset_particle + cloth.nx1 - 1);
			if (anchors.size() != K) return;

			std::vector<std::vector<LRAEntry>> best(V, std::vector<LRAEntry>(K));

			for (uint32_t aGlobal : anchors) {
				int aLocal = globalToLocal[aGlobal];

				auto dist = Dijkstra(V, adj, aLocal);
				for (int li = 0; li < V; ++li) {
					InsertBestK(best[li], aGlobal, dist[li]);
				}
			}

			float slack = 0.1f;
			for (int li = 0; li < V; ++li) {
				uint32_t gi = verts[li];
				for (uint32_t k = 0; k < K; ++k) {
					uint32_t slot = gi * K + k;
					if (best[li][k].anchor == 0xFFFFFFFFu || !std::isfinite(best[li][k].r)) {
						lra_ids[slot] = 0xFFFFFFFFu;
						lra_r[slot] = 0.0f;
					}
					else {
						lra_ids[slot] = best[li][k].anchor;
						lra_r[slot] = best[li][k].r * (1.0f + slack);
					}
				}
			}
		}
	}

	void ResetConstraints(std::vector<glm::vec4>& positions, std::vector<uint32_t>& indices)
	{
		// Edge
		{
			for (auto& edge : edges) {
				int i = edge.i;
				int j = edge.j;

				glm::vec3 pi = glm::vec3(positions[i]);
				glm::vec3 pj = glm::vec3(positions[j]);
				float rest = glm::length(pj - pi);
				edge.rest = rest;
				edge.lambda = 0.0f;
			}
		}

		// Shear
		{
			for (int i = 0; i < num_shears; i++)
			{
				shears[i].lambda = 0.0f;
			}

		}

		// Bend
		{
			for (int i = 0; i < num_bends; i++)
			{
				bends[i].lambda = 0.0f;
			}
		}

		// Area
		{
			for (int i = 0; i < num_areas; i++)
			{
				areas[i].lambda = 0.0f;
			}
		}

	}

	struct SSBO {
		vk::raii::Buffer delta_x{ nullptr };
		vk::raii::Buffer delta_y{ nullptr };
		vk::raii::Buffer delta_z{ nullptr };
		vk::raii::Buffer delta_count{ nullptr };
		vk::raii::Buffer edge{ nullptr };
		vk::raii::Buffer shear{ nullptr };
		vk::raii::Buffer bend{ nullptr };
		vk::raii::Buffer grab_state{ nullptr };
		vk::raii::Buffer area{ nullptr };
		vk::raii::Buffer volume{ nullptr };
		vk::raii::Buffer collider{ nullptr };
		vk::raii::Buffer delta_v{ nullptr };
		vk::raii::Buffer lra_ids{ nullptr };
		vk::raii::Buffer lra_r{ nullptr };
	} ssbos_;

	struct SSBOMemory {
		vk::raii::DeviceMemory delta_x{ nullptr };
		vk::raii::DeviceMemory delta_y{ nullptr };
		vk::raii::DeviceMemory delta_z{ nullptr };
		vk::raii::DeviceMemory delta_count{ nullptr };
		vk::raii::DeviceMemory edge{ nullptr };
		vk::raii::DeviceMemory shear{ nullptr };
		vk::raii::DeviceMemory bend{ nullptr };
		vk::raii::DeviceMemory grab_state{ nullptr };
		vk::raii::DeviceMemory area{ nullptr };
		vk::raii::DeviceMemory volume{ nullptr };
		vk::raii::DeviceMemory collider{ nullptr };
		vk::raii::DeviceMemory delta_v{ nullptr };
		vk::raii::DeviceMemory lra_ids{ nullptr };
		vk::raii::DeviceMemory lra_r{ nullptr };
	} ssbo_memories_;

	struct SSBOSize {
		vk::DeviceSize delta_x = 0;
		vk::DeviceSize delta_y = 0;
		vk::DeviceSize delta_z = 0;
		vk::DeviceSize delta_count = 0;
		vk::DeviceSize edge = 0;
		vk::DeviceSize shear = 0;
		vk::DeviceSize bend = 0;
		vk::DeviceSize grab_state = 0;
		vk::DeviceSize area = 0;
		vk::DeviceSize volume = 0;
		vk::DeviceSize collider = 0;
		vk::DeviceSize delta_v = 0;
		vk::DeviceSize lra_ids = 0;
		vk::DeviceSize lra_r = 0;
	} ssbo_size_;

	struct Staging {
		vk::raii::Buffer edge{ nullptr };
		vk::raii::Buffer shear{ nullptr };
		vk::raii::Buffer bend{ nullptr };
		vk::raii::Buffer area{ nullptr };
		vk::raii::Buffer collider{ nullptr };
		vk::raii::Buffer volume{ nullptr };
	} staging_;

	struct StagingMemory {
		vk::raii::DeviceMemory edge{ nullptr };
		vk::raii::DeviceMemory shear{ nullptr };
		vk::raii::DeviceMemory bend{ nullptr };
		vk::raii::DeviceMemory area{ nullptr };
		vk::raii::DeviceMemory collider{ nullptr };
		vk::raii::DeviceMemory volume{ nullptr };
	} staging_memories_;

	struct StagingMapped {
		void* edge{ nullptr };
		void* shear{ nullptr };
		void* bend{ nullptr };
		void* area{ nullptr };
		void* collider{ nullptr };
		void* volume{ nullptr };
	} staging_mapped_;
};