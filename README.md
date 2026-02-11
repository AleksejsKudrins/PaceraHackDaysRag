# RAG Hello World UI App

A dummy Retrieval-Augmented Generation (RAG) user interface built with Streamlit.

## Project Structure

- `app/`: Contains the Streamlit application code.
- `docs/`: Project documentation and overviews.
- `research/`: Research papers and PDFs related to RAG.
- `.github/workflows/`: CI/CD pipeline for deployment.
- `Dockerfile` & `docker-compose.yml`: Containerization configuration.
- `DEPLOYMENT.md`: Detailed deployment instructions.

## Getting Started

### Prerequisites

- Docker and Docker Compose

### Running Locally

1. Clone the repository.
2. Run the application:
   ```bash
   docker-compose up --build
   ```
3. Open your browser to `http://localhost:8501`.

## Deployment

Refer to [DEPLOYMENT.md](DEPLOYMENT.md) for instructions on how to deploy this app to a Linux server using GitHub Actions.
