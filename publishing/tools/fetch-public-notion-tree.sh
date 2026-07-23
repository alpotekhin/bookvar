#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <page-id> <output-json>" >&2
  exit 2
fi

page_id="$1"
output_json="$2"
work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

curl -L --max-time 30 -sS \
  'https://www.notion.so/api/v3/loadPageChunk' \
  -H 'content-type: application/json' \
  --data "$(jq -cn --arg page_id "$page_id" \
    '{pageId:$page_id,limit:100,cursor:{stack:[]},chunkNumber:0,verticalColumns:false}')" \
  > "$output_json"

space_id="$(jq -r --arg page_id "$page_id" \
  '.recordMap.block[$page_id].spaceId' "$output_json")"

for _ in {1..20}; do
  jq -c --arg space_id "$space_id" '
    (.recordMap.block | keys | INDEX(.)) as $known
    | {
        requests: [
          .recordMap.block[]
          | (.value.value // .value)
          | (.content // [])[]
          | select($known[.] == null)
          | {
              pointer: {
                table: "block",
                id: .,
                spaceId: $space_id
              },
              version: -1
            }
        ]
      }
  ' "$output_json" > "$work_dir/request.json"

  request_count="$(jq '.requests | length' "$work_dir/request.json")"
  if [[ "$request_count" -eq 0 ]]; then
    exit 0
  fi

  curl -L --max-time 30 -sS \
    'https://www.notion.so/api/v3/syncRecordValues' \
    -H 'content-type: application/json' \
    --data-binary @"$work_dir/request.json" \
    > "$work_dir/response.json"

  jq -s '
    .[0].recordMap.block += (.[1].recordMap.block // {})
    | .[0]
  ' "$output_json" "$work_dir/response.json" > "$work_dir/merged.json"
  mv "$work_dir/merged.json" "$output_json"
done

echo "Notion block tree exceeded 20 fetch rounds" >&2
exit 1
