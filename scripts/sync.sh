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
  # 교체 전에 의존 목록을 기억해 둔다. 지운 뒤에는 비교할 대상이 없다.
  local req_before=""
  [[ -f "$DEST/requirements.txt" ]] && req_before=$(cksum < "$DEST/requirements.txt")
  rm -rf "$DEST"; mkdir -p "$DEST"
  git -C "$STAGING" archive "$tag" | tar -x -C "$DEST"
  printf '%s %s\n' "$tag" "$sha" > "$DEST/VERSION"
  log "tag $tag ($sha)"
  log "$DEST/ replaced ($(find "$DEST" -type f | wc -l | tr -d ' ') files, no .git)"

  # 4. 사내 자산 자리 — DEST 밖이어야 갱신에 살아남는다
  mkdir -p outputs notebooks

  # 5. 사내 설정 — 설정 파일을 쓰는 프로젝트에서만. 있으면 절대 건드리지 않는다
  local ex="$DEST/configs/example.yaml"
  if [[ -f "$ex" ]]; then
    mkdir -p configs
    if [[ ! -f configs/local.yaml ]]; then
      cp "$ex" configs/local.yaml
      log "configs/local.yaml 생성 — 사내 실값을 채우세요"
    else
      log "configs/local.yaml exists — kept"
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

  # 안내 문구용 진입점 — src/run.py 를 우선하고, 없으면 src/ 의 유일한 .py 를 쓴다
  local entry="$DEST/src/run.py"
  if [[ ! -f "$entry" ]]; then
    local pys=("$DEST"/src/*.py)
    if [[ ${#pys[@]} -eq 1 && -f "${pys[0]}" ]]; then entry="${pys[0]}"; else entry="$DEST/src/<entry>.py"; fi
  fi

  # ---------------------------------------------------------------------
  # 안내 문구 — **규격 부록에서 이 블록만 갈라져 있다.**
  #
  # 부록은 매번 pip 를 찍는데 늘 필요한 것이 아니다. 사람은 여기 찍힌 줄을
  # 그대로 따라가므로, 필요 없는 줄이 섞이면 매번 안 해도 될 일을 하거나 -
  # 더 나쁘게는 - 이 안내를 통째로 안 믿게 된다. 사내에서는 화면이 유일한
  # 출력이라 안 믿게 되는 쪽이 더 비싸다.
  # ---------------------------------------------------------------------
  printf '\nnext:\n'
  local req_after=""
  [[ -f "$DEST/requirements.txt" ]] && req_after=$(cksum < "$DEST/requirements.txt")
  if [[ -n "$req_after" && "$req_before" != "$req_after" ]]; then
    if [[ -z "$req_before" ]]; then
      printf '  의존 설치 (최초 1회)\n'
    else
      printf '  ⚠ requirements.txt 가 바뀌었습니다\n'
    fi
    printf '    pip install --dry-run -r %s/requirements.txt && pip check\n' "$DEST"
    printf '    pip install -r %s/requirements.txt      # --upgrade 는 쓰지 않는다\n' "$DEST"
  fi
  printf '  python %s --dry-run                        # ① 합성 스모크\n' "$entry"
  printf '  python %s --conv-data <실데이터> --limit 1000   # ② 계약 확인\n' "$entry"
  printf '  python %s --conv-data <실데이터> --output-dir outputs   # ③ 전체\n' "$entry"

  # 대시보드가 있는 프로젝트면 알려준다. 결과를 본 뒤에 쓰는 것이라 지금 당장
  # 할 일은 아니지만, 알려주지 않으면 알아낼 방법이 없다 - 써보고 실패해야만
  # 의존이 따로 있다는 걸 알게 된다.
  local dash="$DEST/src/dashboard.py"
  if [[ -f "$dash" ]]; then
    printf '\n  대시보드 (선택 · 결과를 본 뒤에)\n'
    [[ -f "$DEST/requirements-dashboard.txt" ]] && \
      printf '    python -m pip install -r %s/requirements-dashboard.txt\n' "$DEST"
    printf '    python -m streamlit run %s -- --result outputs/conv_parsed.json\n' "$dash"
  fi
}

main "$@"
