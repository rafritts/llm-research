#!/bin/bash

# Check if Supabase CLI is installed
if ! command -v supabase &> /dev/null; then
    echo "Installing Supabase CLI..."
    curl -Ls https://cli.supabase.com/install.sh | sh
fi

# Initialize Supabase project
npx supabase init --force

# Start Supabase
npx supabase start

# Enable pgvector extension and create documents table
npx supabase db psql << EOF
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id bigserial PRIMARY KEY,
    content text,
    embedding vector(768),
    metadata jsonb,
    created_at timestamp with time zone DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(768),
    match_count int DEFAULT 5
) RETURNS TABLE (
    content text,
    metadata jsonb
) language plpgsql
as $$
begin
    return query
    select documents.content, documents.metadata
    from documents
    order by documents.embedding <-> query_embedding
    limit match_count;
end;
$$;
EOF

# Get URL and anon key
SUPABASE_URL=$(npx supabase status | grep 'API URL' | awk '{print $NF}')
SUPABASE_KEY=$(npx supabase status | grep 'anon key' | awk '{print $NF}')

# Export environment variables
echo "export SUPABASE_URL=$SUPABASE_URL" > .env
echo "export SUPABASE_KEY=$SUPABASE_KEY" >> .env

echo "Supabase setup complete! Environment variables saved to .env"
echo "Please source the .env file: source .env"