#include "geometry_generator.h"

// ---------------------------------------------------------
// Square
// ---------------------------------------------------------
MeshData GeometryGenerator::MakeSquare(float scale)
{
    MeshData meshData;

    meshData.vertices = {
        // front (Z+를 정면, normal = -Z로 가정한 기존 코드 유지)
        Vertex{ glm::vec3(-scale, -scale, 0.0f), glm::vec2(1.0f, 1.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec4(0.0f) },
        Vertex{ glm::vec3(-scale,  scale, 0.0f), glm::vec2(1.0f, 0.0f),   glm::vec3(0.0f, 0.0f, -1.0f), glm::vec4(0.0f) },
        Vertex{ glm::vec3(scale,  scale, 0.0f), glm::vec2(0.0f, 0.0f),  glm::vec3(0.0f, 0.0f, -1.0f), glm::vec4(0.0f) },
        Vertex{ glm::vec3(scale, -scale, 0.0f), glm::vec2(0.0f, 1.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec4(0.0f) },
    };

    meshData.indices = { 0, 1, 2, 0, 2, 3 };
    meshData.indices_count = meshData.indices.size();

    CalculateTangents(meshData);
    return meshData;
}

// ---------------------------------------------------------
// Box
// ---------------------------------------------------------
MeshData GeometryGenerator::MakeBox(float scale)
{
    MeshData meshData;

    meshData.vertices = {
        // front (-Z)
        { glm::vec3(-scale, -scale, -scale), glm::vec2(1.0f, 1.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec4(0.0f) },
        { glm::vec3(-scale,  scale, -scale), glm::vec2(1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec4(0.0f) },
        { glm::vec3(scale,  scale, -scale), glm::vec2(0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec4(0.0f) },
        { glm::vec3(scale, -scale, -scale), glm::vec2(0.0f, 1.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec4(0.0f) },

        // back (+Z)
        { glm::vec3(scale, -scale,  scale), glm::vec2(1.0f, 1.0f), glm::vec3(0.0f, 0.0f,  1.0f), glm::vec4(0.0f) },
        { glm::vec3(scale,  scale,  scale), glm::vec2(1.0f, 0.0f), glm::vec3(0.0f, 0.0f,  1.0f), glm::vec4(0.0f) },
        { glm::vec3(-scale,  scale,  scale), glm::vec2(0.0f, 0.0f), glm::vec3(0.0f, 0.0f,  1.0f), glm::vec4(0.0f) },
        { glm::vec3(-scale, -scale,  scale), glm::vec2(0.0f, 1.0f), glm::vec3(0.0f, 0.0f,  1.0f), glm::vec4(0.0f) },

        // top (+Y)
        { glm::vec3(-scale,  scale, -scale), glm::vec2(1.0f, 1.0f), glm::vec3(0.0f,  1.0f, 0.0f), glm::vec4(0.0f) },
        { glm::vec3(-scale,  scale,  scale), glm::vec2(1.0f, 0.0f), glm::vec3(0.0f,  1.0f, 0.0f), glm::vec4(0.0f) },
        { glm::vec3(scale,  scale,  scale), glm::vec2(0.0f, 0.0f), glm::vec3(0.0f,  1.0f, 0.0f), glm::vec4(0.0f) },
        { glm::vec3(scale,  scale, -scale), glm::vec2(0.0f, 1.0f), glm::vec3(0.0f,  1.0f, 0.0f), glm::vec4(0.0f) },

        // bottom (-Y)
        { glm::vec3(-scale, -scale,  scale), glm::vec2(1.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec4(0.0f) },
        { glm::vec3(-scale, -scale, -scale), glm::vec2(1.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec4(0.0f) },
        { glm::vec3(scale, -scale, -scale), glm::vec2(0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec4(0.0f) },
        { glm::vec3(scale, -scale,  scale), glm::vec2(0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec4(0.0f) },

        // left (-X)
        { glm::vec3(-scale, -scale,  scale), glm::vec2(1.0f, 1.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec4(0.0f) },
        { glm::vec3(-scale,  scale,  scale), glm::vec2(1.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec4(0.0f) },
        { glm::vec3(-scale,  scale, -scale), glm::vec2(0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec4(0.0f) },
        { glm::vec3(-scale, -scale, -scale), glm::vec2(0.0f, 1.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec4(0.0f) },

        // right (+X)
        { glm::vec3(scale, -scale, -scale), glm::vec2(1.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec4(0.0f) },
        { glm::vec3(scale,  scale, -scale), glm::vec2(1.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec4(0.0f) },
        { glm::vec3(scale,  scale,  scale), glm::vec2(0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec4(0.0f) },
        { glm::vec3(scale, -scale,  scale), glm::vec2(0.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec4(0.0f) },
    };

    meshData.indices = {
        0, 1, 2, 0, 2, 3,
        4, 5, 6, 4, 6, 7,
        8, 9, 10, 8, 10, 11,
        12, 13, 14, 12, 14, 15,
        16, 17, 18, 16, 18, 19,
        20, 21, 22, 20, 22, 23,
    };
    meshData.indices_count = meshData.indices.size();

    CalculateTangents(meshData);
    return meshData;
}

// ---------------------------------------------------------
// Bounds Box (라인 박스)
// ---------------------------------------------------------
MeshData GeometryGenerator::MakeBoundsBox()
{
    MeshData meshData;

    meshData.vertices = {
        { glm::vec3(-1.0f, -1.0f, -1.0f), glm::vec2(0.0f, 1.0f), glm::vec3(0,0,-1), glm::vec4(0.0f) },
        { glm::vec3(-1.0f,  1.0f, -1.0f), glm::vec2(0.0f, 0.0f), glm::vec3(0,0,-1), glm::vec4(0.0f) },
        { glm::vec3(1.0f,  1.0f, -1.0f), glm::vec2(1.0f, 0.0f), glm::vec3(0,0,-1), glm::vec4(0.0f) },
        { glm::vec3(1.0f, -1.0f, -1.0f), glm::vec2(1.0f, 1.0f), glm::vec3(0,0,-1), glm::vec4(0.0f) },

        { glm::vec3(1.0f, -1.0f,  1.0f), glm::vec2(0.0f, 1.0f), glm::vec3(0,0, 1), glm::vec4(0.0f) },
        { glm::vec3(1.0f,  1.0f,  1.0f), glm::vec2(0.0f, 0.0f), glm::vec3(0,0, 1), glm::vec4(0.0f) },
        { glm::vec3(-1.0f,  1.0f,  1.0f), glm::vec2(1.0f, 0.0f), glm::vec3(0,0, 1), glm::vec4(0.0f) },
        { glm::vec3(-1.0f, -1.0f,  1.0f), glm::vec2(1.0f, 1.0f), glm::vec3(0,0, 1), glm::vec4(0.0f) },
    };

    meshData.indices = {
        0, 1, 1, 2, 2, 3, 3, 0,
        4, 5, 5, 6, 6, 7, 7, 4,
        0, 7, 1, 6, 2, 5, 3, 4,
    };
    meshData.indices_count = meshData.indices.size();

    return meshData;
}

// ---------------------------------------------------------
// Cylinder (측면만, top/bottom cap은 없음)
// ---------------------------------------------------------
MeshData GeometryGenerator::MakeCylinder(
    float bottomRadius, float topRadius,
    float height, int sliceCount)
{
    MeshData meshData;
    auto& vertices = meshData.vertices;

    float halfH = 0.5f * height;
    float dTheta = -glm::two_pi<float>() / static_cast<float>(sliceCount);

    // bottom ring
    for (int i = 0; i <= sliceCount; ++i)
    {
        float theta = dTheta * static_cast<float>(i);
        float c = std::cos(theta);
        float s = std::sin(theta);

        Vertex v{};
        v.pos = glm::vec3(bottomRadius * c, -halfH, bottomRadius * s);
        v.uv = glm::vec2(static_cast<float>(i) / sliceCount, 1.0f);
        v.normal = glm::normalize(glm::vec3(c, 0.0f, s)); // 단순 radial normal
        v.tangent = glm::vec4(0.0f); // CalculateTangents에서 다시 채움

        vertices.push_back(v);
    }

    // top ring
    for (int i = 0; i <= sliceCount; ++i)
    {
        float theta = dTheta * static_cast<float>(i);
        float c = std::cos(theta);
        float s = std::sin(theta);

        Vertex v{};
        v.pos = glm::vec3(topRadius * c, +halfH, topRadius * s);
        v.uv = glm::vec2(static_cast<float>(i) / sliceCount, 0.0f);
        v.normal = glm::normalize(glm::vec3(c, 0.0f, s));
        v.tangent = glm::vec4(0.0f);

        vertices.push_back(v);
    }

    auto& indices = meshData.indices;

    for (int i = 0; i < sliceCount; ++i)
    {
        indices.push_back(i);
        indices.push_back(i + sliceCount + 1);
        indices.push_back(i + sliceCount + 2);

        indices.push_back(i);
        indices.push_back(i + sliceCount + 2);
        indices.push_back(i + 1);
    }
    meshData.indices_count = meshData.indices.size();

    CalculateTangents(meshData);
    return meshData;
}

// ---------------------------------------------------------
// Sphere
// ---------------------------------------------------------
MeshData GeometryGenerator::MakeSphere(
    float radius,
    int numSlices, int numStacks,
    const glm::vec2 texScale)
{
    MeshData meshData;
    auto& vertices = meshData.vertices;
    auto& indices = meshData.indices;

    // φ: 0..π (stack)
    // θ: 0..2π (slice)
    for (int j = 0; j <= numStacks; ++j)
    {
        float phi = glm::pi<float>() * static_cast<float>(j) / static_cast<float>(numStacks);

        float sinPhi = std::sin(phi);
        float cosPhi = std::cos(phi);

        for (int i = 0; i <= numSlices; ++i)
        {
            float theta = glm::two_pi<float>() * static_cast<float>(i) / static_cast<float>(numSlices);

            float sinTheta = std::sin(theta);
            float cosTheta = std::cos(theta);

            glm::vec3 pos(
                radius * sinPhi * cosTheta,
                radius * cosPhi,
                radius * sinPhi * sinTheta
            );

            Vertex v{};
            v.pos = pos;
            v.normal = glm::normalize(pos);

            v.uv = glm::vec2(
                (static_cast<float>(i) / numSlices) * texScale.x,
                1.0f - (static_cast<float>(j) / numStacks) * texScale.y
            );

            // θ 방향으로의 미분을 이용한 대략적인 tangent
            glm::vec3 tangent(
                -radius * sinPhi * sinTheta,
                0.0f,
                radius * sinPhi * cosTheta
            );
            if (glm::length2(tangent) > 0.0f)
                tangent = glm::normalize(tangent);
            v.tangent = glm::vec4(tangent, 1.0f);

            vertices.push_back(v);
        }
    }

    for (int j = 0; j < numStacks; ++j)
    {
        int rowStart = (numSlices + 1) * j;
        for (int i = 0; i < numSlices; ++i)
        {
            indices.push_back(rowStart + i);
            indices.push_back(rowStart + i + numSlices + 1);
            indices.push_back(rowStart + i + numSlices + 2);

            indices.push_back(rowStart + i);
            indices.push_back(rowStart + i + numSlices + 2);
            indices.push_back(rowStart + i + 1);
        }
    }
    meshData.indices_count = meshData.indices.size();

    // 원하는 경우 여기서 CalculateTangents(meshData)를 다시 호출해서
    // UV 기반으로 정확한 tangent를 덮어쓸 수 있음.
    // CalculateTangents(meshData);

    return meshData;
}

// ---------------------------------------------------------
// Tangent 계산 (GLM 버전)
// ---------------------------------------------------------
void GeometryGenerator::CalculateTangents(MeshData& meshData)
{
    std::vector<glm::vec3> accumulatedTangents(meshData.vertices.size(), glm::vec3(0.0f));

    // 삼각형마다 tangent 누적
    for (size_t i = 0; i + 2 < meshData.indices.size(); i += 3)
    {
        uint32_t i0 = meshData.indices[i];
        uint32_t i1 = meshData.indices[i + 1];
        uint32_t i2 = meshData.indices[i + 2];

        Vertex& v0 = meshData.vertices[i0];
        Vertex& v1 = meshData.vertices[i1];
        Vertex& v2 = meshData.vertices[i2];

        const glm::vec3& p0 = v0.pos;
        const glm::vec3& p1 = v1.pos;
        const glm::vec3& p2 = v2.pos;

        const glm::vec2& uv0 = v0.uv;
        const glm::vec2& uv1 = v1.uv;
        const glm::vec2& uv2 = v2.uv;

        glm::vec3 edge1 = p1 - p0;
        glm::vec3 edge2 = p2 - p0;

        glm::vec2 dUV1 = uv1 - uv0;
        glm::vec2 dUV2 = uv2 - uv0;

        float denom = dUV1.x * dUV2.y - dUV1.y * dUV2.x;
        if (std::abs(denom) < 1e-8f)
            continue;

        float r = 1.0f / denom;

        glm::vec3 tangent = (edge1 * dUV2.y - edge2 * dUV1.y) * r;

        accumulatedTangents[i0] += tangent;
        accumulatedTangents[i1] += tangent;
        accumulatedTangents[i2] += tangent;
    }

    // 정규화 후 Vertex.tangent에 기록
    for (size_t i = 0; i < meshData.vertices.size(); ++i)
    {
        glm::vec3 t = accumulatedTangents[i];
        if (glm::length2(t) > 0.0f)
            t = glm::normalize(t);

        meshData.vertices[i].tangent = t;
    }
}
