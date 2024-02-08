""" PaperPlane: a very simple, flat-file, static blog generator.
Created on Sat Feb 21 2015
Author: Leo Isikdogan
"""
import codecs
import glob
import os
import re
import unicodedata

import dateutil.parser
import jinja2
import markdown
import yaml


class Page:
    def __init__(self, markdown_file):
        self.content_tags = ["media", "link", "tag"]
        self._read_markdown(markdown_file)
        self._split_markdown_to_sections()
        self._parse_markdown_content()
        self._embed_videos()
        self._embed_tags()

    def _read_markdown(self, markdown_file):
        with codecs.open(markdown_file, "r", "utf-8") as f:
            self.title = f.readline()
            f.readline()  # skip a line
            self.content = f.read()

    def _split_markdown_to_sections(self):
        self.content_sections = []
        for x in self.content.split("^^^"):
            lines = x.strip().splitlines()
            title = lines[0].strip("#").strip()
            content = "\n".join(lines[1:]).strip()
            self.content_sections.append({"title": title, "content": content})

    def _parse_markdown_content(self):
        extensions = ["markdown.extensions.extra"]
        self.content = markdown.markdown(self.content, extensions=extensions)
        for content_section in self.content_sections:
            content_section["content"] = markdown.markdown(
                content_section["content"], extensions=extensions
            )

    def get_slugified_title(self, title):
        slugs = unicode(title, "utf8") if not isinstance(title, str) else title
        slugs = slugs.replace("\u0131", "i")
        slugs = unicodedata.normalize("NFKD", slugs).encode("ascii", "ignore").decode("ascii")
        slugs = re.sub("[^\w\s-]", "", slugs).strip().lower()
        return re.sub("[-\s]+", "-", slugs)

    @staticmethod
    def parse_youtube_url(url):
        youtube_regex = (
            r"(https?://)?(www\.)?"
            "(youtube|youtu|youtube-nocookie)\.(com|be)/"
            "(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})"
        )
        youtube_regex_match = re.match(youtube_regex, url)
        if youtube_regex_match:
            return youtube_regex_match.group(6)
        return youtube_regex_match

    def _embed_videos(self):
        matches = re.finditer("<p>\[vid\](.*?)\[/vid\]</p>", self.content)
        for match in matches:
            vidcode = self.parse_youtube_url(match.group(1))
            if vidcode != None:
                embed_code = (
                    '<div class="ratio ratio-16x9 mb-4">'
                    '<iframe src="https://www.youtube.com/embed/{}?" allowfullscreen></iframe>'
                    "</div>"
                ).format(vidcode)
                self.content = self.content.replace(match.group(0), embed_code)

    def _embed_tags(self):
        def replace_tag(section, pattern, tag):
            for match in re.finditer(pattern, section["content"]):
                section["content"] = section["content"].replace(match.group(0), "")
                section[tag] = match.group(1)

        for section in self.content_sections:
            for tag in self.content_tags:
                replace_tag(section, r"<p>\[{}\](.*?)\[/{}\]</p>".format(tag, tag), tag)
                replace_tag(section, r"\[{}\](.*?)\[/{}\]".format(tag, tag), tag)

    def get_dictionary(self):
        return self.__dict__


class TemplateRenderer:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader("templates"))

    @classmethod
    def create_html(cls, filename, template, **kwargs):
        template = cls.env.get_template(template)
        html_file = template.render(kwargs)
        with open(filename, "wb") as f:
            f.write(html_file.encode("utf8"))


class Leetcode(Page):
    def __init__(self, markdown_file):
        super().__init__(markdown_file)

    def create_html_page(self, **kwargs):
        TemplateRenderer.create_html(
            "../public/index.html",
            "leetcode_template.html",
            post=self.get_dictionary(),
            subdir="",
            **kwargs
        )


if __name__ == "__main__":
    leetcode = Leetcode("content.md")
    leetcode.create_html_page()
