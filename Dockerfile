FROM rust:1-slim as builder

# Set the working directory
WORKDIR /usr/src/app

# Copy the Cargo files and source code from the rust_impl directory
COPY rust_impl/Cargo.toml rust_impl/Cargo.lock ./
COPY rust_impl/src ./src

# Build the release binary
RUN cargo build --release

# Final stage
FROM debian:bullseye-slim as final

# Set the working directory
WORKDIR /app

# Copy the shared datasets from the build context's root
COPY datasets/ ./datasets/

# Copy the compiled binary from the builder stage
COPY --from=builder /usr/src/app/target/release/rust_impl .

# Set the entrypoint to run the Rust binary
ENTRYPOINT ["./rust_impl"]