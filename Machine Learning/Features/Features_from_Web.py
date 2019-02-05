import user_agents

ua = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/56.0.2924.76 Chrome/56.0.2924.76 Safari/537.36'
ua = user_agents.parse(ua)

print('Is a bot?', ua.is_bot)
print('Is mobile?', ua.is_mobile)
print('Is PC?', ua.is_pc)
print('OS Family', ua.os.family)
print('OS Version?', ua.os.version)
print('Browser Family?', ua.browser.family)
print('Browser Version', ua.browser.version)
