# 포트폴리오 사이트 (nimnusmik.github.io)

단일 `index.html` 정적 사이트. master에 푸시하면 GitHub Pages가 1~2분 내 자동 배포.

## 구조 ("Follow the Data" 스크롤 스토리)
- 히어로 → ch01 Semo → ch02 Global Vision → ch03 Projects(취업용 대표 4개) → ch04 Side Projects(관심사) → ch05 USC → finale
- 본문 폭 960px(`.wrap`). 문단에 별도 max-width 걸지 말 것.
- 카드: `.grid > a.card` (tag / h3 / p / .stack). 새 카드 추가 시 `.stagger.visible > *:nth-child(n)` 딜레이도 개수만큼 확인.

## 업데이트 규칙
- **취업용 프로젝트(ch03)**: EDGAR 애널리스트 같은 대표작만. 4개 유지가 기본, 교체 우선.
- **사이드프로젝트(ch04)**: 라이브 제품(이달아, yzrecipe 등). 수익/사이드허슬 언급 금지, 기술 중심 영어 카피.
- 카드 링크: 공개 GitHub 레포 또는 라이브 URL만 (푸시 전 `gh repo view`로 공개 여부 확인).
- 회사(Global Vision) 관련: 거래처명·매출·실데이터 절대 금지, 기술 스택과 역할만.
- 언어: 사이트는 영어, 커밋 메시지는 한국어.
- 이력서 변경 반영: 히어로 lede(학위·목표 인턴십), ch05 USC 문단, 스킬 칩 동기화.

## 배포 전 확인
로컬 `python3 -m http.server`로 한 번 렌더링 확인 후 커밋·푸시.
