# 開発TODO

- [ ] テスト整備を再開する際に、既存`tests/`のpackage importと旧CLI呼び出しをscripts構成へ移行する。
  - ユーザー指示により、現時点ではテストの変更・追加・実行を保留する。
  - テストtaskは移行が完了するまで定義しない。

- [x] `afterok`による全部品の自動統合を実装する。
  - `submit`はtarget wall IDSの全コンポーネントをジョブとして投入する。
  - 全ジョブの正常終了後に`pixi run merge`を実行し、各結果を対応する`grid_subset`へ保存する。
  - 直接実行の`render`は一つのコンポーネントを計算し、その結果を含むwall IDSを保存する。
- [ ] HPC上で部品ジョブと`afterok`統合ジョブの実投入を確認する。
