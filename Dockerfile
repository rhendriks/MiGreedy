# base image with cargo chef (cache dependencies and faster runtime
FROM rust:1.85-slim AS chef
RUN cargo install cargo-chef
WORKDIR /app

# create recipe of the dependencies
FROM chef AS planner
COPY rust_impl/Cargo.toml ./ 
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo chef prepare --recipe-path recipe.json

# building
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# build dependencies only
RUN cargo chef cook --release --recipe-path recipe.json

# copy source and build
COPY rust_impl/ .
RUN cargo build --release

# runtime
FROM debian:bookworm-slim AS final
WORKDIR /app

# copy datasets and binary
COPY datasets/ ./datasets/
COPY --from=builder /app/target/release/migreedy .

ENTRYPOINT ["./migreedy"]
