#!/usr/bin/env bash
#
# BB → AA 이식 스크립트. **AA 루트에서** 실행한다.
#
#   bash .staging/{BB}/scripts/sync.sh <tag>
#
# {AA}·{BB} 의 실제 이름은 프로젝트마다 다르다. {BB} 는 이 스크립트의 위치에서 유도하고
# ({AA}/.staging/{BB}/scripts/sync.sh), {AA} 는 실행 위치(cwd)라 이름이 필요 없다.
#
# 최초 1회든 갱신이든 같은 명령이며, 몇 번을 돌려도 같은 상태가 된다.
# 이 스크립트는 실행 도중 checkout 으로 자기 자신을 바꾸므로, 본문 전체를
# main() 으로 감싸 파싱이 먼저 끝나게 한다. (bash 는 스크립트를 조금씩 읽어가며 실행한다)

set -euo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)   # {AA}/.staging/{BB}/scripts
REPO_DIR=$(dirname "$SELF_DIR")                             # {AA}/.staging/{BB}
NAME=$(basename "$REPO_DIR")                                # {BB}
STAGING=".staging/$NAME"
DEST="$NAME"
DATA_EXT='csv|tsv|parquet|xlsx|xls|pkl|pickle|npy|npz|h5|feather|sqlite'

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

main() {
  local tag="${1:-}"
  [[ -n "$tag" ]] || die "태그를 지정하세요:  bash $STAGING/scripts/sync.sh <tag>"
  [[ "$REPO_DIR" == "$(pwd -P)/.staging/$NAME" ]] || die \
    "AA 루트에서 실행하세요. 기대 위치: <AA>/.staging/$NAME/scripts/sync.sh, 현재 cwd: $(pwd -P)"
  [[ -d "$STAGING/.git" ]] || die "$STAGING 이 clone 이 아닙니다 (.git 없음)"

  # 1. 안전장치 — 커밋보다 먼저 깔아둔다
  [[ -f .staging/.gitignore ]] || printf '*\n' > .staging/.gitignore
  if [[ ! -f .gitignore ]] || ! grep -qx '\.staging/' .gitignore; then
    printf '.staging/\n' >> .gitignore
    log "AA/.gitignore 에 .staging/ 추가"
  fi

  # 2. 태그 확보 — 태그 없이는 실행하지 않는다
  git -C "$STAGING" fetch --tags --quiet
  if ! git -C "$STAGING" rev-parse -q --verify "refs/tags/$tag^{}" >/dev/null; then
    warn "태그 '$tag' 가 없습니다. 사용 가능한 태그:"
    git -C "$STAGING" tag -l >&2
    exit 1
  fi
  git -C "$STAGING" -c advice.detachedHead=false checkout --quiet "$tag"
  local sha; sha=$(git -C "$STAGING" rev-parse --short HEAD)

  # 3. 실행 사본 통째 교체
  [[ ! -e "$DEST/.git" ]] || die "$DEST 에 .git 이 있습니다. clone 인지 확인하고 직접 정리하세요 (자동 삭제하지 않습니다)"
  rm -rf "$DEST"; mkdir -p "$DEST"
  git -C "$STAGING" archive "$tag" | tar -x -C "$DEST"
  printf '%s %s\n' "$tag" "$sha" > "$DEST/VERSION"
  log "tag $tag ($sha)"
  log "$DEST/ replaced ($(find "$DEST" -type f | wc -l | tr -d ' ') files, no .git)"

  # 4. 사내 자산 자리 — DEST 밖이어야 갱신에 살아남는다
  mkdir -p configs outputs notebooks

  # 5. 사내 설정 — 있으면 절대 건드리지 않는다
  local ex="$DEST/configs/example.yaml"
  if [[ ! -f configs/local.yaml ]]; then
    [[ -f "$ex" ]] || die "$ex 이 없습니다"
    cp "$ex" configs/local.yaml
    log "configs/local.yaml 생성 — 사내 실값을 채우세요"
  else
    log "configs/local.yaml exists — kept"
    if [[ -f "$ex" ]]; then
      local missing
      missing=$(comm -23 <(yaml_keys "$ex") <(yaml_keys configs/local.yaml) | tr '\n' ' ')
      missing="${missing%"${missing##*[! ]}"}"
      [[ -z "$missing" ]] || warn "example.yaml 에만 있는 키: $missing"
    fi
  fi

  # 6. 유출 점검 — 하나라도 걸리면 실패로 끝낸다
  [[ ! -e "$DEST/.git" ]] || die "leak check: $DEST/.git 이 존재합니다"
  local leaked
  leaked=$(find "$DEST" -type f | grep -Ei "\.($DATA_EXT)\$" || true)
  if [[ -n "$leaked" ]]; then
    warn "데이터 파일이 사본에 있습니다:"; printf '%s\n' "$leaked" >&2
    rm -rf "$DEST"   # 실수로 커밋되는 것을 막기 위해 사본을 남기지 않는다
    die "leak check FAILED — $DEST 를 제거했습니다. BB 의 .gitignore 를 고치고 새 태그를 내세요"
  fi
  log "leak check: OK"

  # 안내 문구용 패키지 이름 — src/ 아래 디렉터리가 하나면 그것으로 본다
  local pkg="<pkg>" cands=("$DEST"/src/*/)
  [[ ${#cands[@]} -eq 1 && -d "${cands[0]}" ]] && pkg=$(basename "${cands[0]}")

  cat <<EOF

next:
  source <venv>/bin/activate
  pip install --dry-run -r $DEST/requirements.txt && pip check
  PYTHONPATH=$DEST/src python -m $pkg --config configs/local.yaml --dry-run
EOF
}

main "$@"
