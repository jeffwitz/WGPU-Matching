struct Params {
    count: u32,
    centers_count: u32,
    samples_per_axis: u32,
    projection_mode: u32,
    origin: vec2<f32>,
    scale: f32,
    ell: f32,
    half_width: f32,
    half_height: f32,
    _a: f32,
    _b: f32,
};
struct PatchResult {
    displacement_support: vec4<f32>,
    gradient: vec4<f32>,
};
struct FieldValue {
    displacement: vec2<f32>,
    dx: vec2<f32>,
    dy: vec2<f32>,
};

@group(0) @binding(0) var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> affine: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> centers: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> coefficients: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> results: array<PatchResult>;
@group(0) @binding(5) var<uniform> p: Params;

fn evaluate(point: vec2<f32>) -> FieldValue {
    let z = (point - p.origin) / p.scale;
    var value = affine[0] + z.x * affine[1] + z.y * affine[2];
    var dx = affine[1];
    var dy = affine[2];
    for (var k = 0u; k < p.centers_count; k++) {
        let delta = z - centers[k];
        let phi = exp(-dot(delta, delta) / (2.0 * p.ell * p.ell));
        value += coefficients[k] * phi;
        let derivative = coefficients[k] * (-phi / (p.ell * p.ell));
        dx += derivative * delta.x;
        dy += derivative * delta.y;
    }
    return FieldValue(value * p.scale, dx, dy);
}

@compute @workgroup_size(64)
fn evaluate_points(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.count) { return; }
    let result = evaluate(points[i]);
    results[i].displacement_support = vec4<f32>(result.displacement, 0.0, 0.0);
    results[i].gradient = vec4<f32>(result.dx.x, result.dy.x, result.dx.y, result.dy.y);
}

@compute @workgroup_size(64)
fn project_patches(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.count) { return; }
    let center = points[i];
    let center_value = evaluate(center);
    let sample_denominator = max(1.0, f32(p.samples_per_axis - 1u));
    let n = p.samples_per_axis;
    var gx = vec2<f32>(0.0);
    var gy = vec2<f32>(0.0);
    var sum_x2 = 0.0;
    var sum_y2 = 0.0;

    if (p.projection_mode == 1u) {
        gx = center_value.dx;
        gy = center_value.dy;
    } else {
        for (var sy = 0u; sy < n; sy++) {
            let ry = -p.half_height + 2.0 * p.half_height * f32(sy) / sample_denominator;
            for (var sx = 0u; sx < n; sx++) {
                let rx = -p.half_width + 2.0 * p.half_width * f32(sx) / sample_denominator;
                let delta = evaluate(center + vec2<f32>(rx, ry)).displacement
                    - center_value.displacement;
                gx += delta * rx;
                gy += delta * ry;
                sum_x2 += rx * rx;
                sum_y2 += ry * ry;
            }
        }
        gx /= max(sum_x2, 1e-12);
        gy /= max(sum_y2, 1e-12);
    }

    var squared_error = 0.0;
    var maximum_error = 0.0;
    for (var sy = 0u; sy < n; sy++) {
        let ry = -p.half_height + 2.0 * p.half_height * f32(sy) / sample_denominator;
        for (var sx = 0u; sx < n; sx++) {
            let rx = -p.half_width + 2.0 * p.half_width * f32(sx) / sample_denominator;
            let actual = evaluate(center + vec2<f32>(rx, ry)).displacement;
            let predicted = center_value.displacement + gx * rx + gy * ry;
            let error = length(actual - predicted);
            squared_error += error * error;
            maximum_error = max(maximum_error, error);
        }
    }

    results[i].displacement_support = vec4<f32>(
        center_value.displacement,
        sqrt(squared_error / f32(n * n)),
        maximum_error,
    );
    results[i].gradient = vec4<f32>(gx.x, gy.x, gx.y, gy.y);
}
