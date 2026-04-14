// ── 1. 1D Heat Equation (explicit Euler, stencil) ──────────────────────────
fn heat_diffusion(u: Tensor, alpha: f64, dx: f64, dt: f64) -> Tensor {
    let nx = 256.0;
    let r  = alpha * dt / (dx * dx);
    let u_next: Tensor = [0.0];
    for i in 1..255 {
        let u_next = r * (u[i+1] - 2.0*u[i] + u[i-1]) + u[i];
    }
    return u_next;
}

// ── 2. Kinetic energy of a particle system ─────────────────────────────────
fn total_kinetic_energy(m: f64, vx: Tensor, vy: Tensor, vz: Tensor) -> f64 {
    let ke  = 0.0;
    let n   = 1024.0;
    for i in 0..1024 {
        let v2 = vx[i]^2.0 + vy[i]^2.0 + vz[i]^2.0;
        let ke = ke + 0.5 * m * v2;
    }
    return ke;
}

// ── 3. Gravitational potential (Newtonian) ─────────────────────────────────
fn gravitational_potential(G: f64, m1: f64, m2: f64, r: f64) -> f64 {
    return -G * m1 * m2 / r;
}

// ── 4. RMS velocity (Maxwell-Boltzmann) ────────────────────────────────────
fn rms_velocity(kb: f64, T: f64, m: f64) -> f64 {
    return sqrt(3.0 * kb * T / m);
}

// ── 5. Lorenz attractor (single step) ─────────────────────────────────────
fn lorenz_step(x: f64, y: f64, z: f64, sigma: f64, rho: f64, beta: f64, dt: f64) -> Tensor {
    let dx = sigma * (y - x) * dt;
    let dy = (x * (rho - z) - y) * dt;
    let dz = (x * y - beta * z) * dt;
    return [x + dx, y + dy, z + dz];
}

// ── 6. Frobenius norm of a flattened matrix ────────────────────────────────
fn frobenius_norm(A: Tensor, n: f64) -> f64 {
    let acc = 0.0;
    for i in 0..1024 {
        let acc = acc + A[i]^2.0;
    }
    return sqrt(acc);
}
