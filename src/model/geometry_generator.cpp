#include "geometry_generator.h"

Mesh GeometryGenerator::MakeSquare(float scale)
{
    Mesh meshData;

    meshData.vertices = {
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

Mesh GeometryGenerator::MakeBox(float scale)
{
    Mesh meshData;

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
        0, 1, 2, 0, 2, 3,   // front
        4, 5, 6, 4, 6, 7,   // back
        8, 9,10, 8,10,11,   // top
        12,13,14,12,14,15,  // bottom
        16,17,18,16,18,19,  // left
        20,21,22,20,22,23,  // right
    };
    meshData.indices_count = meshData.indices.size();

    CalculateTangents(meshData);
    return meshData;
}

Mesh GeometryGenerator::MakeCylinder(
    float bottomRadius, float topRadius,
    float height, int sliceCount)
{
    Mesh meshData;
    auto& vertices = meshData.vertices;
    auto& indices = meshData.indices;

    float halfH = 0.5f * height;
    float dTheta = glm::two_pi<float>() / static_cast<float>(sliceCount);

    // bottom ring (y = -halfH)
    for (int i = 0; i <= sliceCount; ++i)
    {
        float theta = dTheta * static_cast<float>(i);
        float c = std::cos(theta);
        float s = std::sin(theta);

        Vertex v{};
        v.pos = glm::vec3(bottomRadius * c, -halfH, bottomRadius * s);
        v.uv = glm::vec2(1.0f - static_cast<float>(i) / sliceCount, 1.0f);
        v.normal = glm::normalize(glm::vec3(c, 0.0f, s)); // radial
        v.tangent = glm::vec4(0.0f);

        vertices.push_back(v);
    }

    // top ring (y = +halfH)
    int baseTop = static_cast<int>(vertices.size());
    for (int i = 0; i <= sliceCount; ++i)
    {
        float theta = dTheta * static_cast<float>(i);
        float c = std::cos(theta);
        float s = std::sin(theta);

        Vertex v{};
        v.pos = glm::vec3(topRadius * c, +halfH, topRadius * s);
        v.uv = glm::vec2(1.0f - static_cast<float>(i) / sliceCount, 0.0f);
        v.normal = glm::normalize(glm::vec3(c, 0.0f, s));
        v.tangent = glm::vec4(0.0f);

        vertices.push_back(v);
    }

    int ringCount = sliceCount + 1;

    for (int i = 0; i < sliceCount; ++i)
    {
        int i0 = i;
        int i1 = i + ringCount;
        int i2 = i + 1 + ringCount;
        int i3 = i + 1;

        // triangle 1: i0, i1, i2
        indices.push_back(i0);
        indices.push_back(i1);
        indices.push_back(i2);

        // triangle 2: i0, i2, i3
        indices.push_back(i0);
        indices.push_back(i2);
        indices.push_back(i3);
    }

    meshData.indices_count = meshData.indices.size();

    CalculateTangents(meshData);
    return meshData;
}

Mesh GeometryGenerator::MakeCapsule(
    float bottomRadius, float topRadius,
    float height, int sliceCount)
{

    Mesh meshData;
    auto& vertices = meshData.vertices;
    auto& indices = meshData.indices;

    float halfH = 0.5f * height;

    int numStacks = sliceCount;
    int numSlices = sliceCount;
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

            Vertex v{};
            glm::vec3 pos;
            if (phi <= glm::pi<float>() * 0.5f)
            {
                pos = glm::vec3(
                    topRadius * sinPhi * cosTheta,
                    halfH + topRadius * cosPhi,
                    topRadius * sinPhi * sinTheta
                );
                v.pos = pos;
                v.normal = glm::normalize(pos - glm::vec3(0.0f, halfH, 0.0f));
                v.uv = glm::vec2(
                    1.0f - (static_cast<float>(i) / numSlices),
                    (static_cast<float>(j) / numStacks)
                );

                v.tangent = glm::vec4(0.0f);
                vertices.push_back(v);
            }
            
            if (phi > glm::pi<float>() * 0.5f)
            {
                pos = glm::vec3(
                    topRadius * sinPhi * cosTheta,
                    -halfH + topRadius * cosPhi,
                    topRadius * sinPhi * sinTheta
                );
                v.pos = pos;
                v.normal = glm::normalize(pos - glm::vec3(0.0f, -halfH, 0.0f));
                v.uv = glm::vec2(
                    1.0f - (static_cast<float>(i) / numSlices),
                    (static_cast<float>(j) / numStacks)
                );

                v.tangent = glm::vec4(0.0f);
                vertices.push_back(v);
            }

        }
    }

    int rowStride = numSlices + 1;
    for (int j = 0; j < numStacks; ++j)
    {
        int rowStart = j * rowStride;
        int nextRowStart = (j + 1) * rowStride;

        for (int i = 0; i < numSlices; ++i)
        {
            int i0 = rowStart + i;
            int i1 = nextRowStart + i;
            int i2 = nextRowStart + i + 1;
            int i3 = rowStart + i + 1;

            indices.push_back(i0);
            indices.push_back(i2);
            indices.push_back(i1);

            indices.push_back(i0);
            indices.push_back(i3);
            indices.push_back(i2);
        }
    }

    meshData.indices_count = meshData.indices.size();

    CalculateTangents(meshData);
    return meshData;
}

Mesh GeometryGenerator::MakeSphere(
    float radius,
    int numSlices, int numStacks,
    const glm::vec2 texScale)
{
    Mesh meshData;
    auto& vertices = meshData.vertices;
    auto& indices = meshData.indices;

    // ��: 0..�� (stack)
    // ��: 0..2�� (slice)
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
                1.0f - (static_cast<float>(i) / numSlices) * texScale.x,
                (static_cast<float>(j) / numStacks) * texScale.y
            );

            v.tangent = glm::vec4(0.0f);
            vertices.push_back(v);
        }
    }

    int rowStride = numSlices + 1;
    for (int j = 0; j < numStacks; ++j)
    {
        int rowStart = j * rowStride;
        int nextRowStart = (j + 1) * rowStride;

        for (int i = 0; i < numSlices; ++i)
        {
            int i0 = rowStart + i;
            int i1 = nextRowStart + i;
            int i2 = nextRowStart + i + 1;
            int i3 = rowStart + i + 1;

            indices.push_back(i0);
            indices.push_back(i2);
            indices.push_back(i1);

            indices.push_back(i0);
            indices.push_back(i3);
            indices.push_back(i2);
        }
    }

    meshData.indices_count = meshData.indices.size();

    CalculateTangents(meshData);
    return meshData;
}

void GeometryGenerator::CalculateTangents(Mesh& meshData)
{
    std::vector<glm::vec3> accumulatedTangents(meshData.vertices.size(), glm::vec3(0.0f));

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

    for (size_t i = 0; i < meshData.vertices.size(); ++i)
    {
        glm::vec3 t = accumulatedTangents[i];
        if (glm::length2(t) > 0.0f)
            t = glm::normalize(t);

        meshData.vertices[i].tangent = t;
    }
}
