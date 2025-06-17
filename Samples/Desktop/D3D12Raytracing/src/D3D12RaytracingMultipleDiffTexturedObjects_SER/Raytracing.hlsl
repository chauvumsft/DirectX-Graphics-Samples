//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

        
#ifndef RAYTRACING_HLSL
#define USE_VARYING_ARTIFICIAL_WORK          // comment out to disable the test
#define RAYTRACING_HLSL
#define HLSL
//#define SER_WORKLOAD_TEST
#define WORK_LOOP_ITERATIONS_LIGHT   5000         // «baseline»
#define WORK_LOOP_ITERATIONS_HEAVY   (WORK_LOOP_ITERATIONS_LIGHT * 5)  // 5 × heavier
#define RAYS_WITH_HEAVY_WORK_FRACTION 5            // every 5-th ray
// Constants for controlling light sample counts
#define LIGHT_SAMPLES_LIGHT 1
#define LIGHT_SAMPLES_HEAVY 20
#define MAX_BOUNCES_LIGHT 1
#define MAX_BOUNCES_HEAVY 5


#include "RaytracingHlslCompat.h"
using namespace dx;


RaytracingAccelerationStructure Scene : register(t0, space0);
RWTexture2D<float4> RenderTarget : register(u0);
ByteAddressBuffer Indices : register(t1, space0);
StructuredBuffer<Vertex> Vertices : register(t2, space0);

ConstantBuffer<SceneConstantBuffer> g_sceneCB : register(b0);
ConstantBuffer<CubeConstantBuffer> g_cubeCB : register(b1);
 
    
float rand(float3 seed)
{
// Simple hash function
    return frac(sin(dot(seed, float3(12.9898, 78.233, 45.5432))) * 43758.5453);
}

// Load three 16 bit indices from a byte addressed buffer.
uint3 Load3x16BitIndices(uint offsetBytes)
{
    uint3 indices;

    // ByteAdressBuffer loads must be aligned at a 4 byte boundary.
    // Since we need to read three 16 bit indices: { 0, 1, 2 } 
    // aligned at a 4 byte boundary as: { 0 1 } { 2 0 } { 1 2 } { 0 1 } ...
    // we will load 8 bytes (~ 4 indices { a b | c d }) to handle two possible index triplet layouts,
    // based on first index's offsetBytes being aligned at the 4 byte boundary or not:
    //  Aligned:     { 0 1 | 2 - }
    //  Not aligned: { - 0 | 1 2 }
    const uint dwordAlignedOffset = offsetBytes & ~3;    
    const uint2 four16BitIndices = Indices.Load2(dwordAlignedOffset);
 
    // Aligned: { 0 1 | 2 - } => retrieve first three 16bit indices
    if (dwordAlignedOffset == offsetBytes)
    {
        indices.x = four16BitIndices.x & 0xffff;
        indices.y = (four16BitIndices.x >> 16) & 0xffff;
        indices.z = four16BitIndices.y & 0xffff;
    }
    else // Not aligned: { - 0 | 1 2 } => retrieve last three 16bit indices
    {
        indices.x = (four16BitIndices.x >> 16) & 0xffff;
        indices.y = four16BitIndices.y & 0xffff;
        indices.z = (four16BitIndices.y >> 16) & 0xffff;
    }

    return indices;
}


typedef BuiltInTriangleIntersectionAttributes MyAttributes;
    

struct [raypayload] RayPayload
{
    float4 color : write(caller, closesthit, miss) : read(caller);
    uint iterations : write(caller) : read(closesthit);
};
   

// Retrieve hit world position.
float3 HitWorldPosition()
{
    return WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
}

// Retrieve attribute at a hit position interpolated from vertex attributes using the hit's barycentrics.
float3 HitAttribute(float3 vertexAttribute[3], BuiltInTriangleIntersectionAttributes attr)
{
    return vertexAttribute[0] +
        attr.barycentrics.x * (vertexAttribute[1] - vertexAttribute[0]) +
        attr.barycentrics.y * (vertexAttribute[2] - vertexAttribute[0]);
}

// Generate a ray in world space for a camera pixel corresponding to an index from the dispatched 2D grid.
inline void GenerateCameraRay(uint2 index, out float3 origin, out float3 direction)
{
    float2 xy = index + 0.5f; // center in the middle of the pixel.
    float2 screenPos = xy / DispatchRaysDimensions().xy * 2.0 - 1.0;

    // Invert Y for DirectX-style coordinates.
    screenPos.y = -screenPos.y;

    // Unproject the pixel coordinate into a ray.
    //float4 world = mul(float4(screenPos, 0, 1), g_sceneCB.projectionToWorld);
        
    // Switch!
    float4 world = mul(g_sceneCB.projectionToWorld, float4(screenPos, 0, 1));

    world.xyz /= world.w;
    origin = g_sceneCB.cameraPosition.xyz;
    direction = normalize(world.xyz - origin);
}

// Diffuse lighting calculation.
float4 CalculateDiffuseLighting(float3 incidentLightRay, float3 normal, float4 diffuseColor)
{
    float3 hitToLight = normalize(-incidentLightRay);
    float fNDotL = saturate(dot(hitToLight, normal));

    return g_cubeCB.albedo * diffuseColor * fNDotL;
}
    

[shader("raygeneration")]
void MyRaygenShader()
{
    float3 rayDir;
    float3 origin;
    
    // Generate a ray for a camera pixel corresponding to an index from the dispatched 2D grid.
    GenerateCameraRay(DispatchRaysIndex().xy, origin, rayDir);

    // Trace the ray.
    // Set the ray's extents.
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = rayDir;
    // Set TMin to a non-zero small value to avoid aliasing issues due to floating - point errors.
    // TMin should be kept small to prevent missing geometry at close contact areas.
    ray.TMin = 0.001;
    ray.TMax = 10000.0;
    
    
    //uint iterations = (g_cubeCB.materialID == 1) ? WORK_LOOP_ITERATIONS_HEAVY : WORK_LOOP_ITERATIONS_LIGHT;
    //#ifdef USE_VARYING_ARTIFICIAL_WORK
    //    // every Nth ray takes the “heavy” path
    //    if ((DispatchRaysIndex().x % RAYS_WITH_HEAVY_WORK_FRACTION) == 0)
    //        iterations = WORK_LOOP_ITERATIONS_HEAVY;
    //#endif   
        
    bool isHeavy = (g_cubeCB.materialID == 1) && ((DispatchRaysIndex().x % RAYS_WITH_HEAVY_WORK_FRACTION) == 0);
    uint iterations = isHeavy ? WORK_LOOP_ITERATIONS_HEAVY : WORK_LOOP_ITERATIONS_LIGHT;

    RayPayload payload =
    {
        float4(0, 0, 0, 0),
        iterations
    };

    #ifdef SER_WORKLOAD_TEST
        HitObject hit = HitObject::TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);
    
        int sortKey = g_cubeCB.materialID * 10 + DispatchRaysIndex().x % 5;
        //int sortKey = (payload.iterations != WORK_LOOP_ITERATIONS_LIGHT) ? 1 : 0;
        dx::MaybeReorderThread(sortKey, 1);

        HitObject::Invoke(hit, payload);
    #else
        TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);
    #endif
        float4 color = payload.color;

        // Write the raytraced color to the output texture.
        RenderTarget[DispatchRaysIndex().xy] = color;
    }

    
[shader("closesthit")]
void MyClosestHitShader(inout RayPayload payload, in MyAttributes attr)
{
    float3 hitPosition = HitWorldPosition();

    // Get the base index of the triangle's first 16 bit index.
    uint indexSizeInBytes = 2;
    uint indicesPerTriangle = 3;
    uint triangleIndexStride = indicesPerTriangle * indexSizeInBytes;
    uint baseIndex = PrimitiveIndex() * triangleIndexStride;

    // Load up 3 16 bit indices for the triangle.
    const uint3 indices = Load3x16BitIndices(baseIndex);

    // Retrieve corresponding vertex normals for the triangle vertices.
    float3 vertexNormals[3] =
    {
        Vertices[indices[0]].normal,
        Vertices[indices[1]].normal,
        Vertices[indices[2]].normal 
    };

    // Compute the triangle's normal.
    // This is redundant and done for illustration purposes 
    // as all the per-vertex normals are the same and match triangle's normal in this sample. 
    float3 triangleNormal = HitAttribute(vertexNormals, attr);

    float4 sampled = float4(1, 1, 1, 1);
    float4 diffuseScale = float4(1, 1, 1, 1);
    float4 specularColor = float4(0, 0, 0, 0);
    float4 lightMaxing = float4(0, 0, 0, 0);
    float3 incidentLightRay = normalize(hitPosition - g_sceneCB.lightPosition.xyz);
    float3 viewDir = normalize(-WorldRayDirection());
    float3 lightDir = normalize(g_sceneCB.lightPosition.xyz - hitPosition);
    float3 normal = triangleNormal;
    float3 finalColor = float3(0, 0, 0);

    if (g_cubeCB.materialID == 1.0f)
    {
        float3 accumulatedSpecular = float3(0, 0, 0);
        float3 accumulatedDiffuse = float3(0, 0, 0);
        float3 F0 = float3(0.04f, 0.04f, 0.04f); // dielectric base reflectivity
        float roughness = 0.3f;
        float alpha = roughness * roughness;

        for (uint i = 0; i < payload.iterations; ++i)
        {
        // Vary half vector slightly per iteration to simulate microfacet sampling
            float3 jitter = float3(sin(i * 12.9898f), cos(i * 78.233f), sin(i * 37.719f)) * 0.01f;
            float3 halfVec = normalize(viewDir + lightDir + jitter);

            float NdotH = saturate(dot(normal, halfVec));
            float NdotL = saturate(dot(normal, lightDir));
            float NdotV = saturate(dot(normal, viewDir));
            float VdotH = saturate(dot(viewDir, halfVec));

        // GGX Normal Distribution Function
            float alpha2 = alpha * alpha;
            float denom = (NdotH * NdotH) * (alpha2 - 1.0f) + 1.0f;
            float D = alpha2 / (3.14 * denom * denom + 1e-5f);

        // Fresnel-Schlick
            float3 F = F0 + (1.0f - F0) * pow(1.0f - VdotH, 5.0f);

        // Smith Geometry Function
            float k = (roughness + 1.0f) * (roughness + 1.0f) / 8.0f;
            float G_V = NdotV / (NdotV * (1.0f - k) + k);
            float G_L = NdotL / (NdotL * (1.0f - k) + k);
            float G = G_V * G_L;

            float3 specular = (D * F * G) / max(4.0f * NdotL * NdotV, 1e-5f);
            float3 diffuse = sampled.rgb * g_sceneCB.lightDiffuseColor.rgb * NdotL;

            accumulatedSpecular += specular;
            accumulatedDiffuse += diffuse;
        }

        finalColor = (accumulatedDiffuse + accumulatedSpecular) / payload.iterations;
    }
        else
    {
        // Simple diffuse for other materials
        float NdotL = saturate(dot(normal, lightDir));
        finalColor = sampled.rgb * g_sceneCB.lightDiffuseColor.rgb * NdotL;
    }

    payload.color = float4(finalColor, 1.0f);
}

[shader("miss")]
void MyMissShader(inout RayPayload payload)
{
    float4 background = float4(0.0f, 0.2f, 0.4f, 1.0f);
    payload.color = background;
}
    



#endif // RAYTRACING_HLSL