import re
def validate_qr_code(qr_code):
    valid_prefix = "EVENT-SEAT-"
    if qr_code.startswith(valid_prefix):
        seat_info = qr_code[len(valid_prefix):]  # 'EVENT-SEAT-' sonrasını al
        # Daha fazla doğrulama yapabilirsiniz, örneğin format kontrolü
        return True if is_valid_seat_info(seat_info) else False
    else:
        return False

def is_valid_seat_info(seat_info):
    # Burada, sıra ve koltuk numarasının doğru formatta olup olmadığını kontrol eden kodlar bulunur
    # Örneğin: 'A-3', 'B-15' gibi.
    pattern = re.compile("^[A-Z]-\d+$")  # Basit bir regex örneği
    return True if pattern.match(seat_info) else False
