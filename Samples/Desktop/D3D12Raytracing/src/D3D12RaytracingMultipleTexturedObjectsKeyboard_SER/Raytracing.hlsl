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
#define SER_WORKLOAD_TEST
#define WORK_LOOP_ITERATIONS_LIGHT   5000         // «baseline»
#define WORK_LOOP_ITERATIONS_HEAVY   (WORK_LOOP_ITERATIONS_LIGHT * 15)  // 5 × heavier
#define RAYS_WITH_HEAVY_WORK_FRACTION 1            // every 5-th ray
// Constants for controlling light sample counts
#define LIGHT_SAMPLES_LIGHT 1
#define LIGHT_SAMPLES_HEAVY 20
#define MAX_BOUNCES_LIGHT 1
#define MAX_BOUNCES_HEAVY 5
#define CUBE_INSTANCE_COUNT 10201 // or pass this via a constant buffer

#include "RaytracingHlslCompat.h"
using namespace dx;
RaytracingAccelerationStructure Scene : register(t0, space0);
RWTexture2D<float4> RenderTarget : register(u0);
    
ByteAddressBuffer IndicesCube : register(t1, space0);
StructuredBuffer<Vertex> VerticesCube : register(t2, space0);
ByteAddressBuffer IndicesComplex : register(t3, space0);
StructuredBuffer<Vertex> VerticesComplex : register(t4, space0);

ConstantBuffer<SceneConstantBuffer> g_sceneCB : register(b0);
ConstantBuffer<CubeConstantBuffer> g_cubeCB : register(b1);
    
Texture2D<float4> MaterialTexture : register(t5, space0);
SamplerState TextureSampler : register(s0);

    
// Procedural checkerboard pattern function.
float CheckerboardPattern(float2 uv, float scale)
{
    float2 scaledUV = uv * scale;
    float2 intPart;
    modf(scaledUV, intPart);
    float checker = fmod(intPart.x + intPart.y, 2.0f);
    return checker < 1.0f ? 0.0f : 1.0f;
}
    
// TODO: fix this for complex shapes
// Load three 16 bit indices from a byte addressed buffer. 
uint3 Load3x16BitIndices(uint offsetBytes, ByteAddressBuffer baf)
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
    const uint2 four16BitIndices = baf.Load2(dwordAlignedOffset);
 
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
   

    RayPayload payload =
    {
        float4(0, 0, 0, 0),
        //iterations
    };


    if (g_sceneCB.enableSER == 1)
    {
        HitObject hit = HitObject::TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);
        uint materialID = hit.LoadLocalRootTableConstant(16);
        uint hintBits = 1;
        
        dx::MaybeReorderThread(hit);
        HitObject::Invoke(hit, payload);
    }
    else
    {
        TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);
            //HitObject hit = HitObject::TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);
           // uint materialID = hit.LoadLocalRootTableConstant(16);
            //uint hintBits = 1;
        
           // dx::MaybeReorderThread(materialID, 1);
           // HitObject::Invoke(hit, payload);
    }
        
    //#endif
        
    float4 color = payload.color;

    // Write the raytraced color to the output texture.
    RenderTarget[DispatchRaysIndex().xy] = color;
}

    
[shader("closesthit")]
void MyClosestHitShader(inout RayPayload payload, in MyAttributes attr)
{
    bool isComplex = InstanceID() >= CUBE_INSTANCE_COUNT;
    float3 hitPosition = HitWorldPosition();

    // Get the base index of the triangle's first 16 bit index.
    uint indexSizeInBytes = 2;
    uint indicesPerTriangle = 3;
    uint triangleIndexStride = indicesPerTriangle * indexSizeInBytes;
    uint baseIndex = PrimitiveIndex() * triangleIndexStride;
    
    // Load up 3 16 bit indices for the triangle.
    // TODO: fix this for complex shapes
    //const uint3 indices = Load3x16BitIndices(baseIndex);
    
    uint3 indices = isComplex ? Load3x16BitIndices(PrimitiveIndex() * 3 * 2, IndicesComplex) : Load3x16BitIndices(PrimitiveIndex() * 3 * 2, IndicesCube);
        
    // Retrieve corresponding vertex normals for the triangle vertices.
    float3 vertexNormals[3];
    if (isComplex)
    {
        vertexNormals[0] = VerticesComplex[indices.x].normal;
        vertexNormals[1] = VerticesComplex[indices.y].normal;
        vertexNormals[2] = VerticesComplex[indices.z].normal;
    }
    else
    {
        vertexNormals[0] = VerticesCube[indices.x].normal;
        vertexNormals[1] = VerticesCube[indices.y].normal;
        vertexNormals[2] = VerticesCube[indices.z].normal;
    }

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

    if (g_cubeCB.materialID == 0)
    {
  
        // The heavy workload test.
            float2 uv = float2(
            frac(hitPosition.x * 0.5 + 0.5),
            frac(hitPosition.z * 0.5 + 0.5)
        );
        
        // Sample the texture using the procedural UVs

        sampled = MaterialTexture.SampleLevel(TextureSampler, uv, 0);
        
            
        // Calculate lighting with the texture
        float NdotL = saturate(dot(normal, lightDir));
        finalColor = sampled.rgb * g_sceneCB.lightDiffuseColor.rgb * NdotL;
        
        // Artficial workload for the heavy workload test.
        //for (uint i = 0; i < 5000; ++i)
        //{
        //  finalColor = sin(finalColor) + cos(finalColor);
        //}
           
    }  
    else
    {
    // Simple diffuse for other materials
        float NdotL = saturate(dot(normal, lightDir));
        finalColor = sampled.rgb * g_sceneCB.lightDiffuseColor.rgb * NdotL;
    }

    payload.color = sampled * float4(finalColor, 1.0f);
}

[shader("miss")]
void MyMissShader(inout RayPayload payload)
{
    float4 background = float4(0.0f, 0.2f, 0.4f, 1.0f);
    payload.color = background;
}
    

#endif // RAYTRACING_HLSL