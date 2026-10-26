"""Direction B: Điều / Khoản / Điểm parsing of vbpl HTML, selection units and citations."""
from viword.legal import Resolver, html_lines, parse_citation, parse_html, parse_lines, units

DECREE = """<html><head><title>x</title><style>p {}</style></head><body>
<p>CHÍNH PHỦ</p><p>Căn cứ Luật Tổ chức Chính phủ;</p><p>Chương I</p><p>QUY ĐỊNH CHUNG</p>
<p><strong>Điều 1. Phạm vi điều chỉnh</strong></p><p>Nghị định này quy định&nbsp;về đê điều.</p>
<p><strong>Điều 2. Đối tượng áp dụng</strong></p>
<p>1. Cơ quan nhà nước.</p><p>2. Tổ chức, cá nhân sau đây:</p><p>a) Doanh nghiệp;</p><p>b) Hộ kinh doanh;</p>
<p>c) Trường hợp khác.</p><p>Việc áp dụng quy định tại điểm này theo hướng dẫn.</p>
<p>3. Trường hợp điều ước quốc tế có quy định khác thì áp dụng điều ước đó.</p>
<p>Chương II</p><p><strong>Điều 3. Hiệu lực thi hành</strong></p><p>Nghị định này có hiệu lực từ ngày 01/01/2025.</p>
<p>Nơi nhận:</p><p>- Ban Bí thư;</p><p>TM. CHÍNH PHỦ</p><p>1. Không phải khoản.</p></body></html>"""


def test_decree_tree():
    doc = parse_html(DECREE)
    assert [a.number for a in doc.articles] == ["1", "2", "3"]
    a1, a2, a3 = doc.articles
    assert a1.intro == "Nghị định này quy định về đê điều." and a1.part == "Chương I"
    assert [c.number for c in a2.clauses] == [1, 2, 3]
    assert [p.letter for p in a2.clauses[1].points] == ["a", "b", "c"]
    assert a2.clauses[1].points[2].text.endswith("theo hướng dẫn.")  # continuation after the points
    assert a3.part == "Chương II" and not a3.clauses
    assert doc.closing[0] == "Nơi nhận:" and doc.closing[-1] == "1. Không phải khoản."
    assert not doc.warnings


def test_quoted_amendments_are_not_structure():
    lines = ["Điều 1. Sửa đổi, bổ sung một số điều của Nghị định số 54/2017/NĐ-CP",
             "1. Sửa đổi, bổ sung khoản 2 Điều 5 như sau:", "“2. Nội dung mới:", "a) Điểm mới;",
             "b) Điểm mới khác.”.", "2. Bổ sung Điều 6a như sau:", "“Điều 6a. Tên điều mới", "1. Khoản mới.”.",
             "3. Sửa đổi khoản 1 Điều 9 như sau:", '"1. Luật này có hiệu lực từ ngày 01 tháng 8 năm 2024.".',
             "Điều 2. Hiệu lực thi hành", "Nghị định này có hiệu lực từ ngày ký."]
    doc = parse_lines(lines)
    assert [a.number for a in doc.articles] == ["1", "2"]
    a1 = doc.articles[0]
    assert [c.number for c in a1.clauses] == [1, 2, 3]
    assert not a1.clauses[0].points and "b) Điểm mới khác" in a1.clauses[0].text
    assert "Điều 6a. Tên điều mới" in a1.clauses[1].text
    assert not doc.warnings


def test_sequences_decide_clauses_and_points():
    lines = ["Điều 4. Các trường hợp", "Các trường hợp sau đây:", "a) Thứ nhất;", "b) Thứ hai;", "c) Ba;", "d) Bốn;",
             "đ) Năm;", "e) Sáu;", "g) Bảy;", "1. Danh sách bên trong điểm g", "Điều 5. Thủ tục", "1. Nộp hồ sơ.",
             "2. Thẩm định.", "1. Mục con, không phải khoản 3", "a) Điểm của khoản 2", "c) Không liền sau a"]
    a4, a5 = parse_lines(lines).articles
    assert not a4.clauses and a4.intro == "Các trường hợp sau đây:"
    assert [p.letter for p in a4.points] == ["a", "b", "c", "d", "đ", "e", "g"]
    assert a4.points[-1].text.endswith("Danh sách bên trong điểm g")
    assert [c.number for c in a5.clauses] == [1, 2]
    assert "Mục con" in a5.clauses[1].text and [p.letter for p in a5.clauses[1].points] == ["a"]
    assert a5.clauses[1].points[0].text.endswith("Không liền sau a")


def test_a_lost_closing_quote_is_reset_by_the_next_article():
    doc = parse_lines(["Điều 1. A", "1. Sửa đổi như sau:", "“2. Nội dung không đóng ngoặc", "Điều 2. B", "1. Một."])
    assert [a.number for a in doc.articles] == ["1", "2"] and doc.articles[1].clauses[0].text == "Một."
    assert doc.warnings == ["quote reset before Điều 2"]


def test_attached_texts_get_their_own_numbering():
    lines = ["Điều 1. Ban hành kèm theo Quyết định này Quy chế bảo vệ bí mật nhà nước.", "Điều 2. Hiệu lực",
             "Quyết định này có hiệu lực sau 15 ngày.", "Điều 3. Trách nhiệm thi hành", "Các đơn vị chịu trách nhiệm.",
             "QUY CHẾ", "BẢO VỆ BÍ MẬT NHÀ NƯỚC", "(Ban hành kèm theo Quyết định số 1657/2005/QĐ-TTCP)", "Chương I",
             "Điều 1. Phạm vi", "1. Một.", "Điều 2. Đối tượng", "Nội dung.", "Nơi nhận:", "KT. TỔNG THANH TRA",
             "QUY ĐỊNH", "VỀ MẪU BIỂU", "(Ban hành kèm theo Quyết định số 1657/2005/QĐ-TTCP)", "Điều 1. Mẫu", "Nội dung."]
    doc = parse_lines(lines)
    assert [(a.attachment, a.number) for a in doc.articles] == [(0, "1"), (0, "2"), (0, "3"), (1, "1"), (1, "2"),
                                                                 (2, "1")]
    assert doc.articles[2].intro == "Các đơn vị chịu trách nhiệm."  # the name left the last article
    assert [x.title for x in doc.attachments] == ["QUY CHẾ BẢO VỆ BÍ MẬT NHÀ NƯỚC", "QUY ĐỊNH VỀ MẪU BIỂU"]
    assert doc.articles[3].part == "Chương I" and not doc.warnings
    u = [u for u in units(doc, "Quyết định 1657/2005/QĐ-TTCP") if u.attachment == 1][0]
    assert u.text.startswith("Quy chế bảo vệ bí mật nhà nước (kèm theo Quyết định 1657/2005/QĐ-TTCP) Điều 1. Phạm vi.")


def test_a_bare_restart_keeps_its_chapter_and_forms_are_not_units():
    doc = parse_lines(["Điều 1. Ban hành Điều lệ", "Điều 2. Hiệu lực", "Có hiệu lực từ ngày ký.", "Chương I",
                       "QUY ĐỊNH CHUNG", "Điều 1. Phạm vi", "Nội dung.", "Điều 2. ………………………………", "Địa chỉ: ……………"])
    assert [(a.attachment, a.number, a.part) for a in doc.articles] == [(0, "1", ""), (0, "2", ""), (1, "1", "Chương I"),
                                                                         (1, "2", "Chương I")]
    assert [(u.attachment, u.article) for u in units(doc)] == [(0, "1"), (0, "2"), (1, "1")]


def test_a_repeated_body_is_dropped():
    body = ["Điều 1. Phạm vi", "Nội dung một.", "Điều 2. Hiệu lực", "Nội dung hai."]
    doc = parse_lines(["NGHỊ ĐỊNH"] + body + body + ["Nơi nhận:"])
    assert [a.number for a in doc.articles] == ["1", "2"] and doc.warnings == ["repeated body dropped"]


def test_units_carry_their_identifiers():
    doc = parse_html(DECREE)
    us = units(doc, "Nghị định 01/2025/NĐ-CP")
    assert [(u.article, u.clause) for u in us] == [("1", None), ("2", 1), ("2", 2), ("2", 3), ("3", None)]
    assert us[2].text.startswith("Nghị định 01/2025/NĐ-CP Điều 2. Đối tượng áp dụng.\n2. Tổ chức")
    assert us[2].points == ["a", "b", "c"] and "\nb) Hộ kinh doanh;" in us[2].text


def test_html_lines():
    assert html_lines("<p>a<br/>b&nbsp; c</p><script>x</script><td>d</td>") == ["a", "b c", "d"]


def test_parse_citation():
    c = parse_citation("Khoản 4 Điều 10 Nghị định 88/2023/NĐ-CP")
    assert (c.point, c.clause, c.article, c.doc_type, c.number) == (None, 4, "10", "Nghị định", "88/2023/NĐ-CP")
    c = parse_citation("Điểm a Khoản 1 Điều 1 Nghị định số 113/2007/NĐ-CP")
    assert (c.point, c.clause, c.article, c.number) == ("a", 1, "1", "113/2007/NĐ-CP")
    c = parse_citation("Khoản 8 Điều 81 Luật Đất đai 2024")
    assert (c.clause, c.article, c.doc_type, c.name, c.year, c.number) == (8, "81", "Luật", "Đất đai", 2024, "")
    c = parse_citation("Khoản 1 Điều 15 Luật Tổ chức chính quyền địa phương năm 2025")
    assert (c.name, c.year) == ("Tổ chức chính quyền địa phương", 2025)
    c = parse_citation("Điều 17 văn bản 08/2022/TT-BGDĐT")
    assert (c.article, c.number, c.doc_type) == ("17", "08/2022/TT-BGDĐT", "")
    assert parse_citation("Không phải trích dẫn") is None


def test_resolver_by_number_and_by_name():
    meta = [{"id": "177815", "title": "Đất đai", "so_ky_hieu": "31/2024/QH15", "loai_van_ban": "Luật",
             "ngay_ban_hanh": "18/01/2024"},
            {"id": "170509", "title": "Luật Sửa đổi, bổ sung một số điều của Luật Đất đai số 31/2024/QH15",
             "so_ky_hieu": "43/2024/QH15", "loai_van_ban": "Luật", "ngay_ban_hanh": "29/06/2024"},
            {"id": "en1", "title": "Đất đai", "so_ky_hieu": "31/2024/QH15", "loai_van_ban": "Bản dịch văn bản",
             "ngay_ban_hanh": "18/01/2024"},
            {"id": "9", "title": "Luật Thuế giá trị gia tăng", "so_ky_hieu": "48/2024/QH15", "loai_van_ban": "Luật",
             "ngay_ban_hanh": "26/11/2024"}]
    meta += [{"id": "182617", "title": "LUẬT TỔ CHỨC TÒA ÁN NHÂN DÂN SỐ 34/2024/QH15", "so_ky_hieu": "34/2024/QH15",
              "loai_van_ban": "Luật", "ngay_ban_hanh": "24/06/2024"},
             {"id": "168125", "title": "Luật khám bệnh, chữa bệnh 2023 số 15/2023/QH15", "so_ky_hieu": "15/2023/QH15",
              "loai_van_ban": "Luật", "ngay_ban_hanh": "09/01/2023"}]
    r = Resolver(meta)
    assert r.resolve(parse_citation("Điều 94 Luật Tổ chức Tòa án nhân dân 2024"))["id"] == "182617"
    assert r.resolve(parse_citation("Điều 3 Luật Khám bệnh, chữa bệnh 2023"))["id"] == "168125"
    assert r.resolve(parse_citation("Khoản 8 Điều 81 Luật Đất đai 2024"))["id"] == "177815"
    assert r.resolve(parse_citation("Điều 1 Luật số 31/2024/QH15"))["id"] == "177815"
    assert r.resolve(parse_citation("Điều 5 Luật Thuế giá trị gia tăng 2024"))["id"] == "9"
    assert r.resolve(parse_citation("Điều 5 Luật Đất đai 2013")) is None
