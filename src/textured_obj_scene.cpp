#include <gfx/radiance_rate/textured_obj_scene.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <tiny_obj_loader.h>

namespace gfx::radiance_rate {
    namespace {
        struct CpuMesh {
            std::vector<gfx::research::Vertex> vertices;
            std::vector<std::uint32_t> indices;
        };

        float roughness_from_shininess(float ns) {
            if (!(ns > 0.0f)) return 0.55f;
            return std::clamp(std::sqrt(2.0f / (ns + 2.0f)), 0.04f, 1.0f);
        }

        std::filesystem::path resolve_texture_path(const std::filesystem::path& obj_path, std::string_view relative_name) {
            if (relative_name.empty()) return {};
            std::string normalized(relative_name);
            std::replace(normalized.begin(), normalized.end(), '\\', '/');
            const std::filesystem::path relative_path(normalized);
            const std::array<std::filesystem::path, 4> candidates = {
                obj_path.parent_path() / relative_path,
                obj_path.parent_path() / relative_path.filename(),
                obj_path.parent_path() / "textures" / relative_path.filename(),
                std::filesystem::path(normalized)
            };
            for (const auto& candidate : candidates) {
                if (!candidate.empty() && std::filesystem::exists(candidate)) return candidate.lexically_normal();
            }
            return {};
        }
    }

    void TexturedObjScene::clear() {
        materials_.clear();
        submeshes_.clear();
        textures_.clear();
        bounds_min_ = {};
        bounds_max_ = {};
    }

    bool TexturedObjScene::load(const std::filesystem::path& path) {
        clear();

        tinyobj::ObjReaderConfig config;
        config.triangulate = true;
        config.vertex_color = false;

        // Keep mtl_search_path empty here. The tinyobjloader revision used by
        // gfx-research-base derives the material directory from the OBJ path
        // in ParseFromFile(). Supplying mtl_search_path in this revision causes
        // the file-based reader to pass an empty search path to LoadObj(), so
        // sponza.mtl is not found and the scene falls back to untextured white.
        tinyobj::ObjReader reader;
        if (!reader.ParseFromFile(path.string(), config)) {
            std::fprintf(stderr, "Textured OBJ: %s\n", reader.Error().c_str());
            return false;
        }
        if (!reader.Warning().empty()) std::fprintf(stderr, "Textured OBJ warning: %s\n", reader.Warning().c_str());

        const auto& attrib = reader.GetAttrib();
        const auto& source_materials = reader.GetMaterials();
        materials_.resize(std::max<std::size_t>(1, source_materials.size()));
        std::fprintf(stderr, "Textured OBJ: %zu materials from %s\n", source_materials.size(), path.string().c_str());

        std::unordered_map<std::string, int> texture_indices;
        for (std::size_t i = 0; i < source_materials.size(); ++i) {
            const auto& source = source_materials[i];
            Material& material = materials_[i];
            material.base_color = glm::vec3(
                static_cast<float>(source.diffuse[0]),
                static_cast<float>(source.diffuse[1]),
                static_cast<float>(source.diffuse[2])
            );
            if (glm::dot(material.base_color, material.base_color) < 1e-6f) material.base_color = glm::vec3(1.0f);
            material.roughness = roughness_from_shininess(static_cast<float>(source.shininess));

            if (!source.diffuse_texname.empty()) {
                const std::filesystem::path texture_path = resolve_texture_path(path, source.diffuse_texname);
                if (!texture_path.empty()) {
                    const std::string key = texture_path.generic_string();
                    auto found = texture_indices.find(key);
                    if (found != texture_indices.end()) {
                        material.texture_index = found->second;
                    } else {
                        gfx::research::Texture2D texture;
                        if (texture.load(texture_path, true, true)) {
                            const int index = static_cast<int>(textures_.size());
                            textures_.push_back(std::move(texture));
                            texture_indices.emplace(key, index);
                            material.texture_index = index;
                            material.base_color = glm::vec3(1.0f);
                        }
                    }
                }
            }
        }

        std::vector<CpuMesh> cpu_meshes(materials_.size());
        bounds_min_ = glm::vec3(std::numeric_limits<float>::max());
        bounds_max_ = glm::vec3(std::numeric_limits<float>::lowest());

        for (const auto& shape : reader.GetShapes()) {
            std::size_t offset = 0;
            for (std::size_t face = 0; face < shape.mesh.num_face_vertices.size(); ++face) {
                const int face_vertices = shape.mesh.num_face_vertices[face];
                int material_index = face < shape.mesh.material_ids.size() ? shape.mesh.material_ids[face] : -1;
                if (material_index < 0 || static_cast<std::size_t>(material_index) >= cpu_meshes.size()) material_index = 0;
                CpuMesh& destination = cpu_meshes[static_cast<std::size_t>(material_index)];

                for (int vertex_index = 0; vertex_index < face_vertices; ++vertex_index) {
                    const tinyobj::index_t index = shape.mesh.indices[offset + static_cast<std::size_t>(vertex_index)];
                    gfx::research::Vertex vertex{};
                    if (index.vertex_index >= 0) {
                        const std::size_t p = static_cast<std::size_t>(index.vertex_index) * 3;
                        if (p + 2 < attrib.vertices.size()) std::memcpy(&vertex.position, attrib.vertices.data() + p, sizeof(float) * 3);
                    }
                    if (index.normal_index >= 0) {
                        const std::size_t n = static_cast<std::size_t>(index.normal_index) * 3;
                        if (n + 2 < attrib.normals.size()) std::memcpy(&vertex.normal, attrib.normals.data() + n, sizeof(float) * 3);
                    }
                    if (index.texcoord_index >= 0) {
                        const std::size_t t = static_cast<std::size_t>(index.texcoord_index) * 2;
                        if (t + 1 < attrib.texcoords.size()) {
                            vertex.uv.x = attrib.texcoords[t];
                            vertex.uv.y = attrib.texcoords[t + 1];
                        }
                    }
                    destination.indices.push_back(static_cast<std::uint32_t>(destination.vertices.size()));
                    destination.vertices.push_back(vertex);
                    bounds_min_ = glm::min(bounds_min_, vertex.position);
                    bounds_max_ = glm::max(bounds_max_, vertex.position);
                }
                offset += static_cast<std::size_t>(face_vertices);
            }
        }

        for (std::size_t i = 0; i < cpu_meshes.size(); ++i) {
            auto& source = cpu_meshes[i];
            if (source.vertices.empty()) continue;
            Submesh submesh;
            submesh.material_index = static_cast<int>(i);
            submesh.mesh.upload(source.vertices, source.indices);
            submeshes_.push_back(std::move(submesh));
        }

        if (submeshes_.empty()) {
            clear();
            return false;
        }
        std::fprintf(stderr, "Textured OBJ: %zu submeshes, %zu diffuse textures loaded\n", submeshes_.size(), textures_.size());
        return true;
    }

    void TexturedObjScene::draw(gfx::research::Shader& shader) const {
        shader.set_int("uBaseTexture", 0);
        for (const auto& submesh : submeshes_) {
            const Material* material = nullptr;
            if (submesh.material_index >= 0 && static_cast<std::size_t>(submesh.material_index) < materials_.size()) {
                material = &materials_[static_cast<std::size_t>(submesh.material_index)];
            }
            const glm::vec3 base_color = material ? material->base_color : glm::vec3(1.0f);
            const float roughness = material ? material->roughness : 0.55f;
            const int texture_index = material ? material->texture_index : -1;
            shader.set_vec3("uBaseColor", base_color);
            shader.set_float("uRoughness", roughness);
            shader.set_int("uUseBaseTexture", texture_index >= 0 ? 1 : 0);
            if (texture_index >= 0 && static_cast<std::size_t>(texture_index) < textures_.size()) textures_[static_cast<std::size_t>(texture_index)].bind(0);
            submesh.mesh.draw();
        }
        shader.set_int("uUseBaseTexture", 0);
    }
}
