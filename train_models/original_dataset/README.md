# Tập dữ liệu huấn luyện
- NIH ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC - Ngày tải: 01/10/2026
- CheXmask: https://physionet.org/content/chexmask-cxr-segmentation-data/1.0.0/ - Ngày tải: 11/03/2026

# Bộ nhãn được bác sĩ chẩn đoán hình ảnh đánh giá (Expert Labels)
https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest

## Cách cài đặt

### Tập dữ liệu: CheXmask (15.6 Gb file ZIP - )
Vì tải trực tiếp trên website sẽ rất chậm, thậm chí sẽ bị lỗi giữa chừng nên sẽ tiếp cận tải tập dữ liệu này thông qua aria2

Mở **PowerShell** và thực hiện lần lượt các lệnh sau.
#### Bước 1: Tạo thư mục chứa công cụ
```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\Tools" | Out-Null
```

#### Bước 2: Tải aria2 từ GitHub (Phiên bản 1.37.0)
```powershell
Invoke-WebRequest `
    -Uri "https://github.com/aria2/aria2/releases/download/release-1.37.0/aria2-1.37.0-win-64bit-build1.zip" `
    -OutFile "$env:TEMP\aria2.zip"
```

#### Bước 3: Giải nén aria2
```powershell
Expand-Archive `
    -Path "$env:TEMP\aria2.zip" `
    -DestinationPath "$env:USERPROFILE\Tools" `
    -Force
```

#### Bước 4: Xác định đường dẫn và kiểm tra
```powershell
$aria2 = (Get-ChildItem `
    "$env:USERPROFILE\Tools" `
    -Filter "aria2c.exe" `
    -Recurse -File |
    Select-Object -First 1).FullName

if (-not $aria2) {
    throw "Không tìm thấy aria2c.exe trong thư mục Tools."
}

Write-Host "Đường dẫn: $aria2"
& $aria2 --version
```
Dự kiến sẽ thấy thông tin sau:
```powershell
aria2 version 1.37.0
Copyright (C) 2006, 2019 Tatsuhiro Tsujikawa

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

** Configuration **
Enabled Features: Async DNS, BitTorrent, Firefox3 Cookie, GZip, HTTPS, Message Digest, Metalink, XML-RPC, SFTP
Hash Algorithms: sha-1, sha-224, sha-256, sha-384, sha-512, md5, adler32
Libraries: zlib/1.3 expat/2.5.0 sqlite3/3.43.1 GMP/6.3.0 c-ares/1.19.1 libssh2/1.11.0
Compiler: mingw-w64 8.0.0 (alpha) / gcc 10-win32 20220113
  built by  x86_64-pc-linux-gnu
  targeting x86_64-w64-mingw32
  on        Nov 15 2023 11:17:49
System: Windows 6.2 (x86_64) (6.2)

Report bugs to https://github.com/aria2/aria2/issues
Visit https://aria2.github.io/
```

#### Bước 5: Nhập địa chỉ thư mục đường dẫn mong muốn
```powershell
$DOWNLOAD_DIRECTORY = "Đường dẫn lưu tập dữ liệu"
```

Ví dụ: Muốn lưu tập dữ liệu tại đường dẫn "D:\CheXmask" thì nhập như sau:
```powershell
$DOWNLOAD_DIRECTORY = "D:\CheXmask"
```

#### Bước 6: Tạo thư mục lưu tập dữ liệu (Nếu chưa tồn tại)
```powershell
New-Item -ItemType Directory -Force -Path $DOWNLOAD_DIRECTORY | Out-Null
```

#### Bước 7: Tải tệp ZIP

##### Tải toàn bộ tập dữ liệu
````powershell
& $aria2 `
    --max-connection-per-server=8 `
    --split=8 `
    --min-split-size=1M `
    --continue=true `
    --max-tries=0 `
    --retry-wait=10 `
    --dir="$DOWNLOAD_DIRECTORY" `
    --out="CheXmask_v1.0.0.zip" `
    "https://physionet.org/content/chexmask-cxr-segmentation-data/get-zip/1.0.0/"
````

**Giải thích các tham số:**

| Tham số | Ý nghĩa |
|---|---|
| `--max-connection-per-server=8` | Giới hạn tối đa 8 kết nối đến một máy chủ. |
| `--split=8` | Chia tệp thành tối đa 8 phần tải song song. |
| `--min-split-size=1M` | Kích thước phân đoạn tối thiểu là 1 MiB. |
| `--continue=true` | Tiếp tục tải dở nếu máy chủ hỗ trợ. |
| `--auto-file-renaming=false` | Không tự đổi tên tệp khi đã tồn tại. |
| `--max-tries=0` | Thử lại không giới hạn. |
| `--retry-wait=10` | Chờ 10 giây trước lần thử lại. |
| `--file-allocation=none` | Không cấp phát trước dung lượng tệp. |
| `--dir` | Thư mục lưu tệp tải xuống. |
| `--out` | Tên tệp đầu ra. |

> Nếu máy chủ hạn chế số kết nối, giảm `--max-connection-per-server` và `--split` từ `8` xuống `2`.

#### Bước 8: Xác định đường dẫn tệp ZIP

```powershell
# Xác định đường dẫn ZIP
$ZIP_PATH = Join-Path $DOWNLOAD_DIRECTORY "CheXmask_v1.0.0.zip"

# Kiểm tra tệp có tồn tại
if (-not (Test-Path -LiteralPath $ZIP_PATH -PathType Leaf)) {
    throw "Không tìm thấy tệp ZIP: $ZIP_PATH"
}

# Lấy thông tin tệp
$file = Get-Item -LiteralPath $ZIP_PATH

Write-Host "Tên tệp: $($file.Name)"
Write-Host "Dung lượng: $([math]::Round($file.Length / 1GB, 2)) GiB"
```
#### Bước 9: Kiểm tra khả năng đọc danh sách tệp trong ZIP:
```powershell
tar -tf "$ZIP_PATH" > $null

if ($LASTEXITCODE -eq 0) {
    Write-Host "Có thể đọc danh sách tệp trong ZIP."
} else {
    Write-Host "Không đọc được ZIP: tệp có thể bị lỗi hoặc chưa tải xong."
}
```
> Lưu ý: Lệnh tar -tf chỉ kiểm tra khả năng đọc danh sách tệp trong ZIP, không xác minh toàn bộ nội dung. Có thể sử dụng checksum SHA-256 do nguồn phát hành cung cấp để kiểm tra tính toàn vẹn của từng tệp dữ liệu.

#### Bước 10: Giải nén dữ liệu
```powershell
# Đường dẫn thư mục giải nén
$EXTRACT_DIRECTORY = Join-Path $DOWNLOAD_DIRECTORY "CheXmask"

# Tạo thư mục nếu chưa tồn tại
New-Item -ItemType Directory -Force -Path $EXTRACT_DIRECTORY | Out-Null

# Giải nén tệp ZIP
tar -xf "$ZIP_PATH" -C "$EXTRACT_DIRECTORY"

# Kiểm tra kết quả giải nén
if ($LASTEXITCODE -ne 0) {
    throw "Giải nén thất bại: $ZIP_PATH"
}

Write-Host "Đã giải nén vào: $EXTRACT_DIRECTORY"
```

Kết quả: Dữ liệu được giải nén vào thư mục Extracted bên trong đường dẫn $DOWNLOAD_DIRECTORY.

Ví dụ, nếu $DOWNLOAD_DIRECTORY = "D:\CheXmask", thư mục giải nén là D:\CheXmask\Extracted.

#### Bước 11: Loại bỏ Folder trung gian
Vì tập dữ liệu có thêm 1 Folder chứa toàn bộ tập dữ liệu bên trong cho nên sẽ tiến hành loại bỏ bớt 1 tầng Folder

CheXmask/
└── chexmask-database-...-1.0.0/
    ├── DATA_DICTIONARY.md
    ├── README.md
    ├── OriginalResolution/
    └── Preprocessed/

Sau khi chuyển đổi
CheXmask/
    ├── DATA_DICTIONARY.md
    ├── README.md
    ├── OriginalResolution/
    └── Preprocessed/

```powershell
# Lấy danh sách nội dung thư mục CheXmask
$ITEMS = @(Get-ChildItem -LiteralPath $EXTRACT_DIRECTORY -Force)

# Chỉ xử lý khi có đúng một thư mục con
if ($ITEMS.Count -eq 1 -and $ITEMS[0].PSIsContainer) {

    $INNER_DIRECTORY = $ITEMS[0].FullName

    # Xác nhận đây là thư mục dữ liệu CheXmask
    $DICTIONARY_PATH = Join-Path $INNER_DIRECTORY "DATA_DICTIONARY.md"

    if (Test-Path -LiteralPath $DICTIONARY_PATH -PathType Leaf) {

        # Lấy toàn bộ tệp và thư mục bên trong
        $FILES = @(Get-ChildItem -LiteralPath $INNER_DIRECTORY -Force)

        # Kiểm tra xung đột trước khi di chuyển
        foreach ($FILE in $FILES) {
            $DESTINATION = Join-Path $EXTRACT_DIRECTORY $FILE.Name

            if (Test-Path -LiteralPath $DESTINATION) {
                throw "Đích đã tồn tại: $DESTINATION"
            }
        }

        # Chuyển toàn bộ nội dung lên một cấp
        foreach ($FILE in $FILES) {
            Move-Item `
                -LiteralPath $FILE.FullName `
                -Destination $EXTRACT_DIRECTORY `
                -ErrorAction Stop
        }

        # Xóa thư mục trung gian đã rỗng
        Remove-Item -LiteralPath $INNER_DIRECTORY -Force -ErrorAction Stop

        Write-Host "Đã loại bỏ thư mục trung gian."
    }
}
```

Kiểm tra lại quá trình
```powershell
$FINAL_DICTIONARY = Join-Path $EXTRACT_DIRECTORY "DATA_DICTIONARY.md"

if (-not (Test-Path -LiteralPath $FINAL_DICTIONARY -PathType Leaf)) {
    throw "Không tìm thấy DATA_DICTIONARY.md sau giải nén."
}

Write-Host "Đã giải nén thành công vào: $EXTRACT_DIRECTORY"
```

#### Bước 12: Chuyển File ZIP vào bên trong Folder
```powershell
# Xác định đường dẫn ZIP đích
$ZIP_DESTINATION = Join-Path $EXTRACT_DIRECTORY (Split-Path $ZIP_PATH -Leaf)

# Chỉ di chuyển nếu ZIP chưa nằm trong thư mục đích
if ([System.IO.Path]::GetFullPath($ZIP_PATH) -ne
    [System.IO.Path]::GetFullPath($ZIP_DESTINATION)) {

    # Không ghi đè ZIP đã tồn tại
    if (Test-Path -LiteralPath $ZIP_DESTINATION) {
        throw "Tệp ZIP đã tồn tại tại: $ZIP_DESTINATION"
    }

    # Chuyển ZIP vào thư mục CheXmask
    Move-Item `
        -LiteralPath $ZIP_PATH `
        -Destination $ZIP_DESTINATION `
        -ErrorAction Stop

    Write-Host "Đã chuyển ZIP vào: $ZIP_DESTINATION"
}

# Cập nhật đường dẫn ZIP cho các bước tiếp theo
$ZIP_PATH = $ZIP_DESTINATION
```

### Tiếp tục tải khi bị gián đoạn
Nếu quá trình tải bị gián đoạn do mất kết nối Internet hoặc đóng PowerShell:

Mở lại PowerShell.

Thiết lập lại biến $DOWNLOAD_DIRECTORY và $aria2 nếu đã mở phiên PowerShell mới.

Chạy lại lệnh tải ở Bước 6 với cùng thư mục lưu và tên tệp.

Khi mở phiên PowerShell mới, sử dụng:

# Đường dẫn lưu dữ liệu (phải giống lần tải trước)
```powershell
$DOWNLOAD_DIRECTORY = "Đường dẫn lưu tập dữ liệu"

# Xác định lại công cụ aria2
$aria2 = (
    Get-ChildItem "$env:USERPROFILE\Tools" `
        -Filter "aria2c.exe" `
        -Recurse `
        -File |
    Select-Object -First 1
).FullName
```

Lưu ý: Không xóa tệp tải dở hoặc tệp điều khiển .aria2 nếu muốn tiếp tục tải. Khả năng tiếp tục tải phụ thuộc vào hỗ trợ HTTP Range của máy chủ.

### Bộ nhãn dữ liệu: nih-cxr-14
Vì Google Cloud Storage không có mục cho tải trên trang web nên phải tiến hành tải thông qua **Google Cloud CLI** để tải dữ liệu từ Google Cloud Storage về máy tính.
Mở **PowerShell** và thực hiện lần lượt các lệnh sau.

#### Bước 1: Cài Google Cloud CLI

Truy cập [trang cài đặt Google Cloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk), chọn **Windows**, tải trình cài đặt và làm theo hướng dẫn. Sau khi cài, mở cửa sổ PowerShell mới.

#### Bước 2: Kiểm tra phiên bản

```powershell
gcloud --version
```

Nếu PowerShell không nhận diện được `gcloud`, thử:

```powershell
gcloud.cmd --version
```

Nếu cả hai lệnh không hoạt động, thực hiện Bước 3. Nếu đã hoạt động, bỏ qua Bước 3 và Bước 4.

### Bước 3: Kiểm tra đường dẫn cài đặt (chỉ khi không nhận diện được lệnh)

```powershell
# Đường dẫn cài đặt thường gặp khi cài cho người dùng hiện tại
$gcloudPath = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin"

# Kiểm tra tệp thực thi
Test-Path (Join-Path $gcloudPath "gcloud.cmd")
```

- `True`: tìm thấy `gcloud.cmd` tại đường dẫn đã kiểm tra.
- `False`: không tìm thấy ở đường dẫn này; xác định thư mục `bin` thực tế trong nơi bạn đã cài Google Cloud CLI trước khi tiếp tục.

### Bước 4: Bổ sung PATH (chỉ khi cần)

Chạy sau khi Bước 3 trả về `True`:

```powershell
# Đọc PATH người dùng
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")

# Thêm đường dẫn mới nếu chưa tồn tại
if (($userPath -split ";") -notcontains $gcloudPath) {
    $newPath = if ([string]::IsNullOrWhiteSpace($userPath)) {
        $gcloudPath
    } else {
        "$userPath;$gcloudPath"
    }

    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
}

# Cập nhật PATH cho cửa sổ PowerShell hiện tại
if (($env:Path -split ";") -notcontains $gcloudPath) {
    $env:Path += ";$gcloudPath"
}

# Kiểm tra
& (Join-Path $gcloudPath "gcloud.cmd") --version
```

Việc sửa PATH không bắt buộc nếu `gcloud --version` đã hoạt động. Lệnh `gcloud.cmd` cũng không yêu cầu thay đổi PowerShell Execution Policy; **không cần** chạy `Set-ExecutionPolicy` chỉ để sử dụng Google Cloud CLI.

### Bước 5: Xác thực tài khoản

```powershell
gcloud auth login
```

Trình duyệt sẽ mở để bạn đăng nhập Google và cấp quyền. Kiểm tra tài khoản đang hoạt động:

```powershell
gcloud auth list
```

Có thể xem cấu hình hiện tại:

```powershell
gcloud config list
```

### Bước 6: Khai báo đường dẫn tùy chỉnh

```powershell
# Thư mục lưu dữ liệu (có thể thay đổi)
$DOWNLOAD_DIR = "Đường dẫn lưu tập dữ liệu"
```
Ví dụ: Muốn lưu tập dữ liệu tại đường dẫn "D:\CheXmask" thì nhập như sau:

```powershell
$DOWNLOAD_DIRECTORY = "D:\nih-cxr-14"
```

```powershell
# Nguồn tải theo quy trình ban đầu
$SOURCE = "gs://healthai-us/nih-cxr-14"

# Tạo thư mục nếu chưa có
New-Item -ItemType Directory -Force -Path $DOWNLOAD_DIR | Out-Null

Write-Host "Nguồn tải: $SOURCE"
Write-Host "Thư mục lưu: $DOWNLOAD_DIR"
```

### Bước 7: Tải toàn bộ thư mục

```powershell
gcloud storage cp `
    --recursive `
    --no-clobber `
    "$SOURCE" `
    "$DOWNLOAD_DIR\"
```

| Tham số | Ý nghĩa |
|---|---|
| `--recursive` | Sao chép toàn bộ thư mục và các thư mục con |
| `--no-clobber` | Bỏ qua tệp đích đã tồn tại, không ghi đè |
| `$SOURCE` | Đường dẫn dữ liệu trên Google Cloud Storage |
| `$DOWNLOAD_DIR` | Thư mục lưu dữ liệu trên máy |

> `--no-clobber` **không kiểm tra tính toàn vẹn** của các tệp đã tồn tại. Lệnh `cp --recursive` cũng có thể tạo thêm thư mục mang tên nguồn bên trong thư mục đích.


### Bước 9: Kiểm tra dung lượng và số lượng tệp đã tải

```powershell
$stats = Get-ChildItem -LiteralPath $DOWNLOAD_DIR -Recurse -File |
    Measure-Object -Property Length -Sum

Write-Host "Số tệp: $($stats.Count)"
Write-Host "Tổng dung lượng: $([math]::Round($stats.Sum / 1GB, 2)) GiB"
```

Đây chỉ là thống kê dữ liệu đã có trên máy, **không phải** xác minh rằng đã tải đủ và không bị lỗi.

### Bước 10: Tiếp tục tải sau khi bị gián đoạn

Mở lại PowerShell, thiết lập lại `$DOWNLOAD_DIR` và `$SOURCE` như Bước 6, sau đó chạy lại Bước 8. Do `--no-clobber`, các tệp đã tồn tại sẽ không bị tải lại.

Có thể dùng `rsync` để đồng bộ ổn định hơn khi thực hiện lại nhiều lần:

```powershell
gcloud storage rsync `
    "$SOURCE" `
    "$DOWNLOAD_DIR" `
    --recursive
```

`rsync` so sánh nguồn và đích để tải các mục bị thiếu hoặc cần cập nhật. Không thêm `--delete-unmatched-destination-objects` nếu không muốn xóa tệp trong thư mục đích.



## Nguồn

- [CheXmask v1.0.0 – PhysioNet](https://physionet.org/content/chexmask-cxr-segmentation-data/1.0.0/)
- [aria2 v1.37.0 – GitHub Releases](https://github.com/aria2/aria2/releases/tag/release-1.37.0)
- [Cài đặt Google Cloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk)
- [NIH Chest X-ray – Google Cloud](https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest)
- [gcloud storage cp](https://docs.cloud.google.com/sdk/gcloud/reference/storage/cp)
- [gcloud storage rsync](https://docs.cloud.google.com/sdk/gcloud/reference/storage/rsync)
- [Requester Pays](https://docs.cloud.google.com/storage/docs/using-requester-pays)



