# 포트폴리오 사이트 (nimnusmik.github.io)

정적 사이트 (`index.html` + `blog/`). master에 푸시하면 GitHub Pages가 1~2분 내 자동 배포.

## 구조 ("Follow the Data" 스크롤 스토리)
- 히어로 → ch01 Semo → ch02 Global Vision(Onbid·GV TM 카드 2개 포함) → ch03 Projects → ch04 Side Projects(관심사) → ch05 USC → finale
- 상단 `#mininav`는 항상 고정 (스크롤 등장 아님). 블로그 페이지들도 같은 바 복사본을 가짐 — 메뉴 바꾸면 세 파일(index, blog/index, 각 글) 동기화.
- 본문 폭 960px(`.wrap`). 문단에 별도 max-width 걸지 말 것.
- 카드: `.grid > a.card` (tag / h3 / p / .stack). 새 카드 추가 시 `.stagger.visible > *:nth-child(n)` 딜레이도 개수만큼 확인.

## Tech Blog (`blog/`)
- 빌드 도구 없음. 글 1개 = 독립 HTML 1개 (`blog/<slug>.html`), 영어.
- 새 글 절차: ① 기존 글(qda-decision-boundaries.html)을 틀로 복사 ② `blog/index.html`의 `<ul class="posts">` 맨 아래에 `<li>` 추가 (오래된 글이 위, 새 글이 아래) (title/tag/date).
- 글 스타일: 다크 단일 테마, 포폴 토큰(레드·골드) 재사용. 그림은 인라인 SVG+JS.
- 톤: 1인칭 학습 노트 ("내가 이해한 과정"). 과제 답안 통짜 복붙 금지 — 개념 설명 위주로.
- 글 20~30개 넘으면 태그 필터·검색 추가 (순수 JS).

## 업데이트 규칙
- **취업용 프로젝트(ch03)**: 대표작 카드 6개(2열×3) 상한, 교체 우선. 기여가 적은 팀 프로젝트는 카드 대신 아래 `.more` 한 줄 목록에.
- **사이드프로젝트(ch04)**: 라이브 제품(이달아, yzrecipe 등). 수익/사이드허슬 언급 금지, 기술 중심 영어 카피.
- 카드 링크: 공개 GitHub 레포 또는 라이브 URL만 (푸시 전 `gh repo view`로 공개 여부 확인).
- 회사(Global Vision) 관련: 거래처명·매출·실데이터 절대 금지, 기술 스택과 역할만.
- 언어: 사이트는 영어, 커밋 메시지는 한국어.
- 이력서 PDF: `resume.pdf` = ATS_v2에서 전화번호만 뺀 웹용(Desktop의 *_web.pdf). 이력서 바꾸면 이것도 다시 만들어 교체.
- 이력서 변경 반영: 히어로 lede(학위·목표 인턴십), ch05 USC 문단, 스킬 칩 동기화. 직함·숫자는 이력서와 똑같이 — 이력서에서 뺀 미검증 수치(Semo 18%·40%)는 사이트에도 쓰지 말 것.

## 배포 전 확인
로컬 `python3 -m http.server`로 한 번 렌더링 확인 후 커밋·푸시.
