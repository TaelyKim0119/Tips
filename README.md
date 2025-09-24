# 네이버 뉴스 & 유튜브 컨텐츠 수집기

이 프로젝트는 특정 키워드에 대한 네이버 뉴스와 유튜브 컨텐츠를 수집하는 도구입니다.

## 주요 기능

1. 네이버 뉴스 수집
   - 특정 기간 내 뉴스 기사 검색
   - 제목, 언론사, 날짜, 링크 정보 수집

2. 유튜브 동영상 정보 수집
   - 조회수 기준 상위 동영상 검색
   - 제목, 채널명, 조회수, 좋아요 수, 댓글 수 등 수집
   - 특정 날짜 이후 업로드된 영상 필터링

## 설치 방법

필요한 라이브러리 설치:
```bash
pip install requests beautifulsoup4 selenium webdriver-manager pandas google-api-python-client openpyxl
```

## 사용 방법

1. YouTube API 키 준비
   - [Google Cloud Console](https://console.cloud.google.com)에서 YouTube Data API v3 키 발급
   - API 키를 코드에 입력

2. 검색어 설정
   ```python
   query = "검색할 키워드"  # 예: "SKT 유심 유출"
   ```

3. 수집 기간 설정
   - 기본값: 2025년 3월 1일 ~ 현재
   - 필요시 start_date 파라미터 수정

4. 실행 및 결과
   - 수집된 데이터는 엑셀 파일로 저장됨
   - 파일명: YouTube_TOP100_[날짜시간].xlsx

## 주의사항

1. API 사용량 제한
   - YouTube Data API는 일일 할당량이 있으므로 주의
   - 필요시 max_results 값 조정

2. 웹사이트 구조 변경
   - 네이버 뉴스의 경우 웹사이트 구조 변경시 CSS 선택자 업데이트 필요

## 라이선스

MIT License

