# Contributing

이슈에는 운영체제, Python 버전, 입력 방식, 사용한 데이터 범위와 재현 절차를 적어 주세요. 코드 변경은 기존 회차 단위 데이터 분할과 안전한 모델 교체 원칙을 유지해야 합니다.

Pull request를 보내기 전에 다음을 실행하세요.

```powershell
python -m unittest discover -s tests -v
python EMG_AI\storage_integrity.py raw_records
```

새 데이터는 참가자 동의와 공개 권한이 확인된 경우에만 추가하세요.

