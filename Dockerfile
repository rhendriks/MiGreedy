# Build musl static binary
FROM --platform=linux/amd64 rust:1-slim AS builder

# Set the working directory
WORKDIR /usr/src/app

# Install musl-tools for static linking
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        musl-tools

# Install the musl target
RUN rustup target add x86_64-unknown-linux-musl

# Copy the Cargo files and source code from the rust_impl directory
COPY rust_impl/Cargo.toml rust_impl/Cargo.lock ./
COPY rust_impl/src ./src

# Build the release binary for the musl target
RUN cargo build --release --target x86_64-unknown-linux-musl

# Strip the binary to reduce size
RUN strip target/x86_64-unknown-linux-musl/release/migreedy

# Final stage: minimal image
FROM scratch AS final

# Set the working directory
WORKDIR /app

# Copy the shared datasets from the build context's root
COPY datasets/ ./datasets/

# Copy the compiled static binary from the builder stage
COPY --from=builder /usr/src/app/target/x86_64-unknown-linux-musl/release/migreedy .

# Set the entrypoint to run the Rust binary
ENTRYPOINT ["./migreedy"]