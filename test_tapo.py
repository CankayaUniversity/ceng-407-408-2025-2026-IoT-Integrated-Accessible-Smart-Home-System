import asyncio
import os
from dotenv import load_dotenv

# Try to load .env from backend directory if it exists
if os.path.exists('backend/.env'):
    load_dotenv('backend/.env')

async def main():
    try:
        from kasa import Discover
    except ImportError:
        print("Error: python-kasa is not installed. Lütfen 'pip install python-kasa' ile kurun.")
        return

    bulb_host = os.getenv("SMART_BULB_HOST", "").strip()
    username = os.getenv("KASA_USERNAME", "").strip()
    password = os.getenv("KASA_PASSWORD", "").strip()

    print(f"SMART_BULB_HOST: {bulb_host}")
    
    if not bulb_host:
        print("Uyarı: SMART_BULB_HOST çevre değişkeni (IP adresi) bulunamadı. Lütfen backend/.env dosyanızı ayarlayın veya doğrudan kodu güncelleyin.")
        bulb_host = input("Lütfen Tapo P100 cihazının IP adresini girin: ").strip()
        if not bulb_host:
            print("İşlem iptal edildi.")
            return

    discover_kwargs = {}
    if username and password:
        print("Kasa hesabı bilgileri bulundu, kimlik doğrulama kullanılıyor...")
        discover_kwargs["username"] = username
        discover_kwargs["password"] = password
    else:
        print("Kasa hesabı bilgileri yok, kimlik doğrulaması olmadan deneniyor...")
        print("(Not: Yeni Tapo P100 modelleri bir Tapo hesabı ile kimlik doğrulaması gerektirebilir)")

    print(f"\n[{bulb_host}] adresindeki cihaza bağlanılıyor...")

    try:
        device = await Discover.discover_single(bulb_host, **discover_kwargs)
        if device is None:
            print("Cihaz bulunamadı! IP adresini ve cihazın açık olup olmadığını kontrol edin.")
            return
            
        await device.update()
        print(f"\nBağlantı başarılı! Cihaz bulundu: {device.alias}")
        print(f"Cihaz şu an: {'AÇIK' if device.is_on else 'KAPALI'}")
        
        while True:
            action = input("\nNe yapmak istersiniz? (1: Aç, 2: Kapat, q: Çık): ")
            if action == '1':
                print("Cihaz açılıyor...")
                await device.turn_on()
                await asyncio.sleep(0.5)
                await device.update()
                print(f"Güncel durum: {'AÇIK' if device.is_on else 'KAPALI'}")
            elif action == '2':
                print("Cihaz kapatılıyor...")
                await device.turn_off()
                await asyncio.sleep(0.5)
                await device.update()
                print(f"Güncel durum: {'AÇIK' if device.is_on else 'KAPALI'}")
            elif action.lower() == 'q':
                break
            else:
                print("Geçersiz giriş!")
                
    except Exception as e:
        print(f"\nHata oluştu: {e}")
        print("Kimlik doğrulaması hatası alıyorsanız, backend/.env dosyasına KASA_USERNAME ve KASA_PASSWORD bilgilerinizi (Tapo/Kasa hesabınızın) eklediğinizden emin olun.")

if __name__ == "__main__":
    asyncio.run(main())
