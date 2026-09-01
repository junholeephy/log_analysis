#!/usr/bin/env bash
#
# {BB} → {AA} 이식 스크립트. 실행 위치에 따라 두 모드로 동작한다.
#
#   본 머신 ({BB} 저장소 루트에서)   bash scripts/sync.sh <tag>
#       → 태그의 archive 를 임시로 풀어 점검만 한다. push 전에 돌린다
#
#   운영     ({AA} 루트에서)           bash .staging/{BB}/scripts/sync.sh <tag>
#       → 이식(교체·VERSION·디렉터리)을 하고 같은 점검을 한 번 더 한다
#
# {AA}·{BB} 의 실제 이름은 프로젝트마다 다르다. {BB} 는 이 스크립트의 위치에서 유도하고
# ({AA}/.staging/{BB}/scripts/sync.sh), {AA} 는 실행 위치(cwd)라 이름이 필요 없다.
#
# 이 스크립트는 실행 도중 checkout 으로 자기 자신을 바꿀 수 있으므로, 본문 전체를
# main() 으로 감싸 파싱이 먼저 끝나게 한다. (bash 는 스크립트를 조금씩 읽어가며 실행한다)

set -euo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_DIR=$(dirname "$SELF_DIR")
NAME=$(basename "$REPO_DIR")
STAGING=".staging/$NAME"
DEST="$NAME"

DATA_EXT='csv|tsv|parquet|xlsx|xls|pkl|pickle|npy|npz|h5|feather|sqlite'
# 이식 표면에 남아서는 안 되는 것들 — .gitattributes 의 export-ignore 로 빼야 한다
FORBIDDEN=(.git .gitattributes .github .claude .cursor .mcp.json CLAUDE.md AGENTS.md
           tools requirements-dev.txt docs/insights)

log()  { printf '[sync] %s\n' "$*"; }
warn() { printf '[sync] ⚠ %s\n' "$*" >&2; }
die()  { printf '[sync] ✗ %s\n' "$*" >&2; exit 1; }

# YAML 의 키를 점 경로로 뽑는다. 2칸 들여쓰기 매핑을 가정하며, 새 키 알림 용도의 근사치다.
yaml_keys() {
  awk '
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*$/ { next }
    /^[[:space:]]*-/ { next }
    {
      line = $0
      match(line, /^[[:space:]]*/); indent = RLENGTH
      sub(/^[[:space:]]*/, "", line)
      if (line ~ /^[A-Za-z0-9_.-]+[[:space:]]*:/) {
        key = line; sub(/[[:space:]]*:.*/, "", key)
        lvl = int(indent / 2)
        path[lvl] = key
        out = path[0]
        for (i = 1; i <= lvl; i++) out = out "." path[i]
        print out
      }
    }
  ' "$1" | sort -u
}

# 트리 안을 훑되 자기 자신(scripts/sync.sh)은 제외하고, 경로를 트리 기준 상대 경로로 줄인다.
scan() {  # scan <dir> <regex>
  grep -rInE "$2" "$1" 2>/dev/null | grep -v "^$1/scripts/sync\.sh:" | sed "s|^$1/||" || true
}

# 이식 표면 점검. 인자로 받은 디렉터리는 "실제로 운영 환경에 도착할 것"이어야 한다.
# 두 모드가 이 함수를 공유하므로 검사 기준이 한 벌뿐이다.
inspect_tree() {
  local d="$1" bad=0 hits f

  for f in "${FORBIDDEN[@]}"; do
    if [[ -e "$d/$f" ]]; then
      warn "이식 표면에 남아있음: $f   → .gitattributes 에 '$f export-ignore' 추가"
      bad=1
    fi
  done

  hits=$(find "$d" -type f | grep -Ei "\.($DATA_EXT)\$" | sed "s|^$d/||" || true)
  if [[ -n "$hits" ]]; then
    warn "데이터 파일:"; printf '%s\n' "$hits" >&2; bad=1
  fi

  hits=$(scan "$d" '^[[:space:]]*(import|from)[[:space:]]+(anthropic|openai)')
  if [[ -n "$hits" ]]; then
    warn "운영 환경에서 쓸 수 없는 API import (C8):"; printf '%s\n' "$hits" >&2; bad=1
  fi

  if [[ -f "$d/requirements.txt" ]]; then
    hits=$(grep -inE '^[[:space:]]*(anthropic|openai|claude)' "$d/requirements.txt" || true)
    if [[ -n "$hits" ]]; then
      warn "requirements.txt 에 개발 전용 패키지:"; printf '%s\n' "$hits" >&2; bad=1
    fi
  fi

  hits=$(scan "$d" '/(Users|home)/[A-Za-z0-9._-]+')
  if [[ -n "$hits" ]]; then
    warn "개인 머신 절대 경로 (§1.3 위반이기도 하다 — 인자로 빼라):"; printf '%s\n' "$hits" >&2; bad=1
  fi

  hits=$(scan "$d" 'Co-Authored-By|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
  if [[ -n "$hits" ]]; then
    warn "이메일·커밋 트레일러:"; printf '%s\n' "$hits" >&2; bad=1
  fi

  return $bad
}

require_tag() {
  local repo="$1" tag="$2"
  if ! git -C "$repo" rev-parse -q --verify "refs/tags/$tag^{}" >/dev/null; then
    warn "태그 '$tag' 가 없습니다. 사용 가능한 태그:"
    git -C "$repo" tag -l >&2
    exit 1
  fi
}

# 본 머신 — archive 결과를 임시로 풀어 점검만 한다
preflight() {
  local tag="$1"
  require_tag "$REPO_DIR" "$tag"
  local sha; sha=$(git -C "$REPO_DIR" rev-parse --short "$tag^{}")
  local tmp; tmp=$(mktemp -d)
  # 값을 지금 확정해 둔다 — 함수를 벗어난 뒤 트랩이 돌 때 $tmp 는 이미 사라지고 없다
  trap "rm -rf '$tmp'" EXIT

  git -C "$REPO_DIR" archive "$tag" | tar -x -C "$tmp"
  log "preflight: $tag ($sha) — $(find "$tmp" -type f | wc -l | tr -d ' ') files"

  if ! inspect_tree "$tmp"; then
    die "preflight FAILED — 위 항목을 고치고 태그를 다시 내세요"
  fi
  log "preflight: OK"
  cat <<EOF

next:
  git push origin $tag
EOF
}

# 운영 환경 — 이식하고 같은 점검을 한 번 더 한다
sync_into_aa() {
  local tag="$1"

  [[ -f .staging/.gitignore ]] || printf '*\n' > .staging/.gitignore
  if [[ ! -f .gitignore ]] || ! grep -qx '\.staging/' .gitignore; then
    printf '.staging/\n' >> .gitignore
    log "$(basename "$(pwd -P)")/.gitignore 에 .staging/ 추가"
  fi

  git -C "$STAGING" fetch --tags --quiet
  require_tag "$STAGING" "$tag"
  git -C "$STAGING" -c advice.detachedHead=false checkout --quiet "$tag"
  local sha; sha=$(git -C "$STAGING" rev-parse --short HEAD)

  [[ ! -e "$DEST/.git" ]] || die "$DEST 에 .git 이 있습니다. clone 인지 확인하고 직접 정리하세요 (자동 삭제하지 않습니다)"
  rm -rf "$DEST"; mkdir -p "$DEST"
  git -C "$STAGING" archive "$tag" | tar -x -C "$DEST"
  printf '%s %s\n' "$tag" "$sha" > "$DEST/VERSION"
  log "tag $tag ($sha)"
  log "$DEST/ replaced ($(find "$DEST" -type f | wc -l | tr -d ' ') files)"

  mkdir -p outputs notebooks

  local ex="$DEST/configs/example.yaml"
  if [[ -f "$ex" ]]; then
    mkdir -p configs
    if [[ ! -f configs/local.yaml ]]; then
      cp "$ex" configs/local.yaml
      log "configs/local.yaml 생성 — 운영 실값을 채우세요"
    else
      log "configs/local.yaml exists — kept"
      local missing
      missing=$(comm -23 <(yaml_keys "$ex") <(yaml_keys configs/local.yaml) | tr '\n' ' ')
      missing="${missing%"${missing##*[! ]}"}"
      [[ -z "$missing" ]] || warn "example.yaml 에만 있는 키: $missing"
    fi
  fi

  if ! inspect_tree "$DEST"; then
    rm -rf "$DEST"   # 실수로 커밋되는 것을 막기 위해 사본을 남기지 않는다
    die "점검 FAILED — $DEST 를 제거했습니다. 본 머신에서 고치고 새 태그를 내세요"
  fi
  log "점검: OK"

  local entry="$DEST/src/run.py"
  if [[ ! -f "$entry" ]]; then
    local pys=("$DEST"/src/*.py)
    if [[ ${#pys[@]} -eq 1 && -f "${pys[0]}" ]]; then entry="${pys[0]}"; else entry="$DEST/src/<entry>.py"; fi
  fi

  cat <<EOF

next:
  source <venv>/bin/activate
  pip install --dry-run -r $DEST/requirements.txt && pip check
  python $entry --dry-run
EOF
}

main() {
  local tag="${1:-}" cwd; cwd=$(pwd -P)
  [[ -n "$tag" ]] || die "태그를 지정하세요:  bash <이 스크립트> <tag>"

  if [[ "$REPO_DIR" == "$cwd" ]]; then
    preflight "$tag"
  elif [[ "$REPO_DIR" == "$cwd/.staging/$NAME" ]]; then
    [[ -d "$STAGING/.git" ]] || die "$STAGING 이 clone 이 아닙니다 (.git 없음)"
    sync_into_aa "$tag"
  else
    die "실행 위치가 맞지 않습니다. 둘 중 하나여야 합니다:
       본 머신:  cd <{BB} 저장소> && bash scripts/sync.sh <tag>
       운영:     cd <{AA}>        && bash .staging/$NAME/scripts/sync.sh <tag>
     현재 cwd: $cwd"
  fi
}

main "$@"
